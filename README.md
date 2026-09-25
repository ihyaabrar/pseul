# PSEUL

**Prediction-landmark-aware feature selection for medical machine learning.**

A feature can be strongly predictive and clinically unusable: it may be recorded
*because* the outcome happened, or only exist *after* the moment the prediction is
meant to be made. Ordinary feature selection cannot tell, because both kinds of
variable are strongly associated with the label. PSEUL makes prediction-time
availability and label proximity explicit terms of the selection, and reports
the discrimination that enforcing them costs instead of hiding it in a headline
metric.

PSEUL scores every candidate on five dimensions, in the order the name spells them:

| | dimension | measures |
|---|---|---|
| **P** | predictive utility | permutation AUC loss |
| **S** | SHAP stability | attribution consistency across folds |
| **E** | evidence relevance | literature support and topical similarity |
| **U** | clinical utility | intervention, stratification, feasibility |
| **L** | leakage risk | how close the feature sits to the label |

Two outputs answer two different questions:

- **PSEUL-Audit** — what drives raw discrimination, ranked with no constraints.
- **PSEUL-Select** — what survives the feature-safety contract: a landmark gate and
  a semantic veto as hard controls, then a utility gate, a soft leakage penalty,
  redundancy control and a score floor.

## Install

```bash
pip install "pseul[lightgbm] @ git+https://github.com/ihyaabrar/pseul.git"
```

Python 3.10 or later. PSEUL accepts any scikit-learn-compatible classifier; the
`lightgbm` extra installs the learner the study used.

## Quickstart

```python
from lightgbm import LGBMClassifier
from pseul import PSEUL, PseulConfig, PseulFeatureProfile

registry = {name: PseulFeatureProfile() for name in X.columns}
registry["DIAGNOSIS_CODE"] = PseulFeatureProfile(label_derived_risk=0.95,
                                                 definitional_overlap=0.95)
registry["FOLLOWUP_VISITS"] = PseulFeatureProfile(available_at_prediction=False)

selector = PSEUL(estimator=LGBMClassifier(verbose=-1),
                 clinical_profile=registry,
                 config=PseulConfig(top_k=5))
selector.fit(X_train, y_train)

selector.selected_features_                    # PSEUL-Select
selector.select_features(mode="audit", top_k=5)  # PSEUL-Audit
selector.summary()                             # every score, per feature
X_small = selector.transform(X_test)
```

[`examples/quickstart.py`](examples/quickstart.py) runs this end to end on the
breast-cancer data with two leaky features added. Its output:

```
PSEUL-Select: ['WORST_SMOOTHNESS', 'MEAN_TEXTURE', 'MEAN_RADIUS', 'SYMMETRY_ERROR', 'CONCAVITY_ERROR']
PSEUL-Audit (ranked without the contract): ['DIAGNOSIS_CODE', 'MEAN_RADIUS', ...]

Audit top-5   held-out AUC 0.997
Select top-5  held-out AUC 0.982
```

The audit ranking puts the label proxy first; PSEUL-Select excludes it and the
post-landmark variable, at a cost of 0.015 AUC that is reported rather than absorbed.

## The registry

Each feature gets a `PseulFeatureProfile`. Every score is in [0, 1].

| field | enters | meaning |
|---|---|---|
| `available_at_prediction` | landmark gate | `False` excludes the feature outright |
| `label_derived_risk`, `definitional_overlap`, `target_proxy_strength` | semantic veto, leakage risk | the veto fires when any of the three reaches `semantic_veto_threshold` (0.85) |
| `evidence`, `topic_similarity` | evidence relevance *E* | blended at η = 0.50 |
| `intervention_availability`, `risk_stratification`, `operational_feasibility` | clinical utility *U* | averaged |
| `interpretation` | — | free text, carried into the summary |

Features without a profile get neutral defaults.

`PseulConfig()` holds the parameters locked for the study: audit weights
α, β, γ, δ = 0.30, 0.20, 0.25, 0.25; veto threshold τ_V = 0.85; score floor
τ_S = 0.20; soft-penalty steepness τ_L = 10; and the rest documented in the class.

## Read this before you use it

**PSEUL does not detect leakage.** It enforces what the registry declares. A label
proxy with neutral ratings is scored like any other feature, and if it predicts
well it will be selected. The study behind this package showed that nulling the
supplied leakage ratings returns the traps to the selected subset. The quality of
the output is the quality of the registry.

**Pass the full candidate pool.** The leakage threshold θ_L is calibrated from the
leakage scores of whatever features you pass to `fit()`. It is relative, not an
absolute safety bound. Removing the most leakage-prone features beforehand makes
θ_L recalibrate downward and can make the remainder look falsely safe.

**The veto threshold is a policy setting.** τ_V = 0.85 is not a calibrated
probability. In the study, one scenario's retained-proxy result held only for
τ_V between 0.80 and 0.90, so a sensitivity sweep over τ_V is worth running on
your own data.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The tests check that the method is the frozen version the study ran and that the
registry's hard controls hold. `pseul/core.py` is a byte-for-byte copy of the
study's `scripts/pseul.py` at commit `f7eb485`; a test fails if it is edited
without updating that record.

## Citing

The paper describing PSEUL is under review. Until it appears, please cite this
repository using [`CITATION.cff`](CITATION.cff).

## Licence

MIT. See [LICENSE](LICENSE).
