"""PSEUL in one screen: two leaky features, and what the registry does with them.

The breast-cancer data has no leakage of its own, so two features are added:

  DIAGNOSIS_CODE   a noisy copy of the label -- the kind of variable that is
                   recorded because the outcome happened, and so predicts it
                   almost perfectly
  FOLLOWUP_VISITS  a variable that only exists after the moment the prediction
                   is meant to be made

An ordinary importance ranking puts DIAGNOSIS_CODE first. PSEUL excludes both,
but only because the registry below says what they are. PSEUL does not detect
leakage on its own: it enforces what the registry declares, and reports what
that enforcement costs.
"""
import warnings

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from pseul import PSEUL, PseulConfig, PseulFeatureProfile

# SHAP announces that its LightGBM binary output is now a list of arrays. PSEUL
# already reads either format and takes the positive class, so the notice is
# informational only.
warnings.filterwarnings("ignore", message="LightGBM binary classifier with TreeExplainer")

rng = np.random.default_rng(0)
data = load_breast_cancer(as_frame=True)
X = data.frame.drop(columns=["target"])
X.columns = [c.replace(" ", "_").upper() for c in X.columns]
y = (data.frame["target"] == 0).astype(int)

# the two leaky features
X["DIAGNOSIS_CODE"] = np.where(rng.random(len(y)) < 0.95, y, 1 - y)
X["FOLLOWUP_VISITS"] = y * rng.poisson(4, len(y)) + rng.poisson(1, len(y))

# the registry: one profile per feature, every score in [0, 1]
registry = {name: PseulFeatureProfile(evidence=0.7, topic_similarity=0.7) for name in X.columns}
registry["DIAGNOSIS_CODE"] = PseulFeatureProfile(
    label_derived_risk=0.95, definitional_overlap=0.95,
    interpretation="recorded because the diagnosis was made",
)
registry["FOLLOWUP_VISITS"] = PseulFeatureProfile(
    available_at_prediction=False,
    interpretation="only exists after the prediction landmark",
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)
model = LGBMClassifier(n_estimators=120, learning_rate=0.05, random_state=42, verbose=-1)

pseul = PSEUL(
    estimator=model,
    clinical_profile=registry,
    config=PseulConfig(n_splits=3, top_k=5, max_shap_samples_per_fold=200),
)
pseul.fit(X_train, y_train)

summary = pseul.summary().set_index("feature")
print("PSEUL-Select:", pseul.selected_features_)
print("PSEUL-Audit (ranked without the contract):",
      pseul.select_features(mode="audit", top_k=5))
print()
print(summary.loc[["DIAGNOSIS_CODE", "FOLLOWUP_VISITS"],
                  ["audit_score", "leakage_risk", "admissible_for_selection", "selected"]])

# what the contract costs, measured on held-out data
audit = pseul.select_features(mode="audit", top_k=5)
for name, cols in (("Audit top-5 ", audit), ("Select top-5", pseul.selected_features_)):
    fitted = LGBMClassifier(n_estimators=120, learning_rate=0.05, random_state=42,
                            verbose=-1).fit(X_train[cols], y_train)
    auc = roc_auc_score(y_test, fitted.predict_proba(X_test[cols])[:, 1])
    print(f"{name}  held-out AUC {auc:.3f}")
