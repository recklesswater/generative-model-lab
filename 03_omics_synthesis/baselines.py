"""Classifiers and the evaluation discipline.

Two decisions here are the ones that make the rest of the experiment meaningful:

1. **The threshold is chosen on out-of-fold training predictions**, never on the test set. The
   operating point is "the most sensitive threshold that still meets a specificity target of
   85%", which is the form a clinical reader actually cares about - an AUC can look fine while
   the thresholded decision is useless.
2. **The external cohort is scored with the same threshold**, not re-calibrated per cohort. If a
   model only works after re-calibration on the target cohort, that is worth knowing.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_model(kind: str = "logreg", seed: int = 0):
    if kind == "logreg":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=0.5, max_iter=5000,
                                                class_weight="balanced"))
    if kind == "gbdt":
        return make_pipeline(StandardScaler(),
                             HistGradientBoostingClassifier(
                                 max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
                                 l2_regularization=1.0, random_state=seed))
    raise KeyError(f"unknown model {kind!r}")


def oof_scores(model, x: np.ndarray, y: np.ndarray, seed: int = 0) -> np.ndarray:
    folds = StratifiedKFold(5, shuffle=True, random_state=seed)
    return cross_val_predict(model, x, y, cv=folds, method="predict_proba")[:, 1]


def operating_point(y: np.ndarray, scores: np.ndarray, target_spec: float = 0.85) -> float:
    """Most sensitive threshold with specificity above ``target_spec``."""
    fpr, tpr, thr = roc_curve(y, scores)
    spec = 1.0 - fpr
    ok = np.where(spec > target_spec)[0]
    if not len(ok):
        return float("inf")
    return float(thr[ok][int(np.argmax(tpr[ok]))])


def metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    return {
        "n": len(y), "auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) > 1 else float("nan"),
        "sensitivity": sens, "specificity": spec,
        "ppv": tp / (tp + fp) if tp + fp else float("nan"),
        "npv": tn / (tn + fn) if tn + fn else float("nan"),
        "threshold": threshold,
    }


def run_one(method: str, x_tr, y_tr, x_test, y_test, x_ext, y_ext,
            model_kind: str = "logreg", seed: int = 0) -> dict:
    """Fit, pick the threshold on out-of-fold training scores, score both held-out sets."""
    model = make_model(model_kind, seed=seed)
    scores_oof = oof_scores(model, x_tr, y_tr, seed=seed)
    threshold = operating_point(y_tr, scores_oof)
    model.fit(x_tr, y_tr)
    internal = metrics(y_test, model.predict_proba(x_test)[:, 1], threshold)
    external = metrics(y_ext, model.predict_proba(x_ext)[:, 1], threshold)
    return {"method": method, "model": model_kind, "seed": seed,
            "train_n": len(y_tr), "internal": internal, "external": external}
