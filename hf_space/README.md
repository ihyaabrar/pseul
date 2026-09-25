---
title: PSEUL
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.28.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: Prediction-landmark-aware feature selection for clinical ML
---

# PSEUL

Prediction-landmark-aware feature selection for medical machine learning.

Upload a CSV with a binary outcome and, optionally, a feature registry. PSEUL
ranks the features without constraints (**PSEUL-Audit**) and selects a subset
that respects the registry (**PSEUL-Select**): variables that do not exist at the
moment of prediction are removed by a landmark gate, and variables rated as
label proxies by a semantic veto.

**PSEUL does not detect leakage.** It enforces what the registry declares, and
reports what that enforcement costs.

This Space samples uploads above 5,000 rows and uses a lighter configuration
(3 folds, 3 permutation repeats) to stay responsive on a free CPU. For full data
and the parameters used in the study, install the package:

```bash
pip install "pseul[lightgbm] @ git+https://github.com/ihyaabrar/pseul.git"
```

Licence: MIT.
