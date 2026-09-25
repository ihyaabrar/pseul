"""What a user relies on: the frozen method is the one the study ran, and the
registry's hard controls hold whatever the data say."""
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier

import pseul
from pseul import PSEUL, PseulConfig, PseulFeatureProfile

warnings.filterwarnings("ignore", message="LightGBM binary classifier with TreeExplainer")


def test_core_is_the_frozen_study_method():
    # core.py is a byte-for-byte copy of the study's scripts/pseul.py; editing
    # it without updating the provenance would break the link to the paper
    digest = hashlib.sha256((Path(pseul.__file__).parent / "core.py").read_bytes()).hexdigest()
    assert digest.startswith(pseul.__source_sha256__)


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(1)
    n = 400
    X = pd.DataFrame({f"X{i}": rng.normal(size=n) for i in range(6)})
    logit = 1.2 * X["X0"] - 0.8 * X["X1"] + 0.5 * X["X2"]
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int))
    X["TRAP"] = np.where(rng.random(n) < 0.95, y, 1 - y)        # near-copy of the label
    X["LATE"] = y * rng.poisson(3, n) + rng.poisson(1, n)       # exists after the landmark
    registry = {c: PseulFeatureProfile() for c in X.columns}
    registry["TRAP"] = PseulFeatureProfile(label_derived_risk=0.95, definitional_overlap=0.95)
    registry["LATE"] = PseulFeatureProfile(available_at_prediction=False)
    model = LGBMClassifier(n_estimators=60, random_state=0, verbose=-1)
    selector = PSEUL(estimator=model, clinical_profile=registry,
                     config=PseulConfig(n_splits=3, top_k=4, max_shap_samples_per_fold=100,
                                        permutation_repeats=2))
    return selector.fit(X, y), X


def test_audit_sees_the_trap(fitted):
    selector, _ = fitted
    assert "TRAP" in selector.select_features(mode="audit", top_k=3)


def test_select_honours_the_registry(fitted):
    selector, _ = fitted
    assert "TRAP" not in selector.selected_features_
    assert "LATE" not in selector.selected_features_
    summary = selector.summary().set_index("feature")
    assert not summary.loc["TRAP", "admissible_for_selection"]
    assert not summary.loc["LATE", "admissible_for_selection"]


def test_transform_returns_the_selected_columns(fitted):
    selector, X = fitted
    out = selector.transform(X)
    assert list(out.columns) == selector.selected_features_
    assert len(out) == len(X)
