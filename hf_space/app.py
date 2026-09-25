"""PSEUL on Hugging Face Spaces: audit a dataset against a feature registry.

The logic is in plain functions so it can be tested without a browser; the
Gradio layer at the bottom only wires them to inputs and outputs.
"""
from __future__ import annotations

# `spaces` has to be imported before anything that could touch CUDA. It exists
# only on Hugging Face's ZeroGPU hardware; everywhere else the app runs without it.
try:
    import spaces
except ImportError:
    spaces = None

import dataclasses
import warnings

import gradio as gr
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.datasets import load_breast_cancer

from pseul import PSEUL, PseulConfig, PseulFeatureProfile

# SHAP's notice that its LightGBM binary output is now a list; PSEUL reads both
# formats and takes the positive class
warnings.filterwarnings("ignore", message="LightGBM binary classifier with TreeExplainer")

MAX_ROWS = 5000          # a free CPU Space; larger uploads are sampled, and say so
PROFILE_FIELDS = {f.name: f.type for f in dataclasses.fields(PseulFeatureProfile)}
SHOWN = ["feature", "selected", "admissible_for_selection", "select_score", "audit_score",
         "predictive_utility", "shap_stability", "clinical_utility", "leakage_risk",
         "pseul_category"]

INTRO = """
# PSEUL

**Prediction-landmark-aware feature selection.** PSEUL ranks features without
constraints (**PSEUL-Audit**), then selects a subset that respects a feature
registry (**PSEUL-Select**): a landmark gate removes variables that do not exist
at prediction time, and a semantic veto removes variables rated as label proxies.

**It does not detect leakage.** It enforces what the registry declares. A label
proxy with neutral ratings is scored like any other feature. The quality of the
output is the quality of the registry.
"""

REGISTRY_HELP = """
**Registry CSV** (optional): one row per feature, a `feature` column, plus any of
`available_at_prediction` (true/false), `label_derived_risk`,
`definitional_overlap`, `target_proxy_strength`, `evidence`, `topic_similarity`,
`intervention_availability`, `risk_stratification`, `operational_feasibility`
(all in [0, 1]). Missing features and missing columns take neutral defaults.
"""


# ------------------------------------------------------------------ logic
def demo_data():
    """Breast-cancer data with a label proxy and a post-landmark variable added."""
    rng = np.random.default_rng(0)
    data = load_breast_cancer(as_frame=True)
    X = data.frame.drop(columns=["target"])
    X.columns = [c.replace(" ", "_").upper() for c in X.columns]
    y = (data.frame["target"] == 0).astype(int)
    X["DIAGNOSIS_CODE"] = np.where(rng.random(len(y)) < 0.95, y, 1 - y)
    X["FOLLOWUP_VISITS"] = y * rng.poisson(4, len(y)) + rng.poisson(1, len(y))
    registry = {c: PseulFeatureProfile(evidence=0.7, topic_similarity=0.7) for c in X.columns}
    registry["DIAGNOSIS_CODE"] = PseulFeatureProfile(
        label_derived_risk=0.95, definitional_overlap=0.95,
        interpretation="recorded because the diagnosis was made")
    registry["FOLLOWUP_VISITS"] = PseulFeatureProfile(
        available_at_prediction=False,
        interpretation="only exists after the prediction landmark")
    return X, y, registry


def parse_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in ("1", "true", "yes", "y", "t")


def parse_registry(table: pd.DataFrame) -> tuple[dict[str, PseulFeatureProfile], list[str]]:
    """Rows to profiles. Returns the profiles and a list of problems found."""
    problems = []
    if "feature" not in table.columns:
        return {}, ["the registry has no `feature` column, so it was ignored"]
    unknown = [c for c in table.columns if c != "feature" and c not in PROFILE_FIELDS]
    if unknown:
        problems.append(f"unrecognised registry columns ignored: {', '.join(unknown)}")
    profiles = {}
    for _, row in table.iterrows():
        kwargs = {}
        for name in PROFILE_FIELDS:
            if name not in table.columns or pd.isna(row[name]):
                continue
            if name == "available_at_prediction":
                kwargs[name] = parse_bool(row[name])
            elif name == "interpretation":
                kwargs[name] = str(row[name])
            else:
                value = float(row[name])
                if not 0.0 <= value <= 1.0:
                    problems.append(f"{row['feature']}: {name}={value} is outside [0, 1]")
                    value = min(max(value, 0.0), 1.0)
                kwargs[name] = value
        profiles[str(row["feature"])] = PseulFeatureProfile(**kwargs)
    return profiles, problems


def prepare(frame: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    notes = []
    y_raw = frame[target]
    if y_raw.nunique(dropna=True) != 2:
        raise gr.Error(f"`{target}` has {y_raw.nunique(dropna=True)} distinct values; "
                       "PSEUL needs a binary outcome.")
    keep = y_raw.notna()
    if (~keep).any():
        notes.append(f"{int((~keep).sum())} rows with a missing outcome were dropped")
    frame, y_raw = frame[keep], y_raw[keep]
    positive = sorted(y_raw.unique(), key=str)[-1]
    y = (y_raw == positive).astype(int).reset_index(drop=True)
    X = frame.drop(columns=[target])
    dropped = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if dropped:
        notes.append(f"non-numeric columns left out (encode them first): {', '.join(dropped[:10])}"
                     + (" ..." if len(dropped) > 10 else ""))
    X = X.drop(columns=dropped).reset_index(drop=True)
    if X.shape[1] < 2:
        raise gr.Error("Fewer than two numeric feature columns remain.")
    if len(X) > MAX_ROWS:
        idx = pd.DataFrame({"y": y}).groupby("y").sample(frac=MAX_ROWS / len(X),
                                                         random_state=42).index
        X, y = X.loc[idx].reset_index(drop=True), y.loc[idx].reset_index(drop=True)
        notes.append(f"a stratified sample of {len(X):,} rows was used; run the package "
                     "locally for the full data")
    notes.append(f"positive class: `{positive}`")
    return X, y, notes


def run_pseul(X, y, registry, top_k: int, tau_v: float):
    config = PseulConfig(n_splits=3, top_k=int(top_k), semantic_veto_threshold=float(tau_v),
                         max_shap_samples_per_fold=200, permutation_repeats=3)
    model = LGBMClassifier(n_estimators=120, learning_rate=0.05, random_state=42, verbose=-1)
    selector = PSEUL(estimator=model, clinical_profile=registry, config=config).fit(X, y)
    summary = selector.summary()
    audit = selector.select_features(mode="audit", top_k=int(top_k))
    select = selector.selected_features_
    excluded = [f for f in audit if f not in select
                and not summary.set_index("feature").loc[f, "admissible_for_selection"]]
    lines = [
        f"**PSEUL-Select** ({len(select)}): " + ", ".join(f"`{f}`" for f in select),
        f"**PSEUL-Audit top {int(top_k)}**: " + ", ".join(f"`{f}`" for f in audit),
    ]
    if excluded:
        lines.append("**In the audit ranking but inadmissible under the registry:** "
                     + ", ".join(f"`{f}`" for f in excluded))
    columns = [c for c in SHOWN if c in summary.columns]
    table = summary[columns].sort_values(["selected", "audit_score"], ascending=[False, False])
    return "\n\n".join(lines), table.round(3)


# ZeroGPU refuses to start a Space with no @spaces.GPU function ("No @spaces.GPU
# function detected during startup"), and free accounts cannot move a new Space
# to CPU hardware. PSEUL needs no GPU, so this function exists only to pass that
# check and is never called: every visitor's run stays on the CPU, with no GPU
# queue and no GPU quota spent.
if spaces is not None:
    @spaces.GPU(duration=1)
    def _zerogpu_startup_check():
        return None


# --------------------------------------------------------------- handlers
def on_demo(top_k, tau_v):
    X, y, registry = demo_data()
    text, table = run_pseul(X, y, registry, top_k, tau_v)
    return ("Demo: breast-cancer data with two leaky features added, `DIAGNOSIS_CODE` "
            "(a label proxy) and `FOLLOWUP_VISITS` (exists only after the landmark).\n\n"
            + text), table


TARGET_NAMES = ("outcome", "target", "label", "y", "class", "diagnosis")


def on_upload(data_file):
    """Offer every column, and preselect one only when it is plainly the outcome:
    binary and named like one. Guessing the last column picked a text column."""
    if data_file is None:
        return gr.update(choices=[], value=None)
    frame = pd.read_csv(data_file, nrows=2000)
    columns = list(frame.columns)
    guess = next((c for c in columns if c.strip().lower() in TARGET_NAMES
                  and frame[c].nunique(dropna=True) == 2), None)
    return gr.update(choices=columns, value=guess)


def on_run(data_file, target, registry_file, top_k, tau_v):
    if data_file is None or not target:
        raise gr.Error("Upload a CSV and choose the outcome column.")
    X, y, notes = prepare(pd.read_csv(data_file), target)
    registry = {}
    if registry_file is not None:
        registry, problems = parse_registry(pd.read_csv(registry_file))
        notes += problems
        missing = [c for c in X.columns if c not in registry]
        if missing:
            notes.append(f"{len(missing)} features have no registry row and use neutral defaults")
    else:
        notes.append("no registry supplied, so every feature is neutral and PSEUL-Select "
                     "applies no landmark or veto exclusions")
    text, table = run_pseul(X, y, registry, top_k, tau_v)
    return text + "\n\n" + "\n".join(f"- {n}" for n in notes), table


# -------------------------------------------------------------------- UI
with gr.Blocks(title="PSEUL") as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        top_k = gr.Slider(2, 20, value=5, step=1, label="subset size (top_k)")
        tau_v = gr.Slider(0.50, 1.00, value=0.85, step=0.05,
                          label="semantic-veto threshold τ_V (a policy setting)")
    with gr.Tab("Demo"):
        demo_button = gr.Button("Run the demo", variant="primary")
    with gr.Tab("Your data"):
        data_file = gr.File(label="data CSV (numeric features, binary outcome)",
                            file_types=[".csv"], type="filepath")
        target = gr.Dropdown(label="outcome column", choices=[])
        gr.Markdown(REGISTRY_HELP)
        registry_file = gr.File(label="registry CSV (optional)", file_types=[".csv"],
                                type="filepath")
        run_button = gr.Button("Run PSEUL", variant="primary")
    result = gr.Markdown()
    table = gr.Dataframe(label="per-feature scores", wrap=True)

    demo_button.click(on_demo, [top_k, tau_v], [result, table])
    data_file.change(on_upload, data_file, target)
    run_button.click(on_run, [data_file, target, registry_file, top_k, tau_v], [result, table])

if __name__ == "__main__":
    demo.launch()
