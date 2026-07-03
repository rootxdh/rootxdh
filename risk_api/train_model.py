"""
Train the Ghost Shopper risk model.

There is no real labelled dataset yet, so we synthesise realistic sessions for
two populations — genuine buyers and "ghost" buyers (fake / joke COD orders) —
and fit a calibrated gradient-boosted classifier. Swap `sample_population()`
for your real order-outcome data (delivered vs RTO) when you have it; the rest
of the pipeline stays the same.

Run:  python train_model.py
Out:  model.joblib
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

from features import FEATURE_NAMES

RNG = np.random.default_rng(42)
MODEL_PATH = "model.joblib"


def sample_population(n: int, ghost: bool) -> np.ndarray:
    """Generate n synthetic sessions. Columns follow FEATURE_NAMES order."""
    if ghost:
        # Ghost buyers: order fast, don't read, rarely pick a size, barely
        # scroll, little mouse movement, often paste, reuse devices.
        time_on_page = RNG.gamma(shape=2.0, scale=2.0, size=n)          # ~1-8s
        desc_opened = RNG.random(n) < 0.10
        size_selected = RNG.random(n) < 0.15
        scroll_depth = RNG.beta(1.5, 6, size=n) * 100                    # low
        mouse_moves = RNG.poisson(3, size=n)
        keystrokes = RNG.poisson(12, size=n)
        pastes = RNG.binomial(2, 0.45, size=n)
        device_reuse = RNG.poisson(1.8, size=n)
    else:
        # Genuine buyers: browse a while, read, pick a size, scroll, move
        # the mouse, type their details, rarely reuse a device.
        time_on_page = RNG.gamma(shape=6.0, scale=8.0, size=n)          # ~20-90s
        desc_opened = RNG.random(n) < 0.80
        size_selected = RNG.random(n) < 0.85
        scroll_depth = RNG.beta(5, 2, size=n) * 100                     # high
        mouse_moves = RNG.poisson(35, size=n)
        keystrokes = RNG.poisson(40, size=n)
        pastes = RNG.binomial(1, 0.10, size=n)
        device_reuse = RNG.poisson(0.2, size=n)

    return np.column_stack([
        time_on_page,
        desc_opened.astype(float),
        size_selected.astype(float),
        scroll_depth,
        mouse_moves.astype(float),
        keystrokes.astype(float),
        pastes.astype(float),
        device_reuse.astype(float),
    ])


def build_dataset(n_each: int = 4000, label_noise: float = 0.08):
    genuine = sample_population(n_each, ghost=False)
    ghost = sample_population(n_each, ghost=True)
    X = np.vstack([genuine, ghost])
    y = np.concatenate([np.zeros(n_each), np.ones(n_each)])  # 1 = ghost

    # Real buyers are not perfectly separable — some genuine customers behave
    # "suspiciously" and some ghosts fake a careful browse. Inject label noise
    # so the calibrated probabilities form a real gradient (and the middle
    # "review" band actually fires) instead of collapsing to 0/1.
    flip = RNG.random(y.shape[0]) < label_noise
    y[flip] = 1 - y[flip]
    return X, y


def main() -> None:
    X, y = build_dataset()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Logistic regression on standardised features gives smooth, well-behaved
    # probabilities — mixed signals land in the middle "review" band instead of
    # collapsing to 0/1. It's also interpretable (each feature has a weight).
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, C=1.0),
    )
    clf.fit(X_tr, y_tr)

    proba = clf.predict_proba(X_te)[:, 1]
    preds = (proba >= 0.5).astype(int)

    print("Features:", FEATURE_NAMES)
    print(f"\nROC-AUC: {roc_auc_score(y_te, proba):.4f}\n")
    print(classification_report(y_te, preds, target_names=["genuine", "ghost"]))

    joblib.dump({"model": clf, "features": FEATURE_NAMES}, MODEL_PATH)
    print(f"Saved -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
