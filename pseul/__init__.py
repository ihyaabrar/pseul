"""PSEUL: prediction-landmark-aware feature selection for medical machine learning.

`core.py` is the frozen method from the study, copied byte for byte rather than
reimplemented, so what this package runs is what the paper reports.
"""
from .core import PSEUL, PseulConfig, PseulFeatureProfile

__all__ = ["PSEUL", "PseulConfig", "PseulFeatureProfile"]

__version__ = "0.1.0"

# provenance of core.py: the study repository at this commit
__source_commit__ = "f7eb485"
__source_sha256__ = "123738752bb86e3b"
