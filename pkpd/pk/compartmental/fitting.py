"""Generic nonlinear least-squares wrapper around scipy.optimize.curve_fit
for any model function in models.py. Weighting + bounds + fit diagnostics."""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import curve_fit

WeightScheme = str  # "uniform" | "inverse_y" | "inverse_y2"


def _weights(y: np.ndarray, scheme: WeightScheme) -> np.ndarray:
    if scheme == "uniform":
        return np.ones_like(y)
    if scheme == "inverse_y":
        return 1.0 / np.clip(y, 1e-12, None)
    if scheme == "inverse_y2":
        return 1.0 / np.clip(y, 1e-12, None) ** 2
    raise ValueError(f"unknown weight scheme: {scheme}")


def fit_model(
    model: Callable,
    t: np.ndarray,
    y: np.ndarray,
    p0: list[float],
    bounds: tuple[list[float], list[float]] | None = None,
    weight_scheme: WeightScheme = "uniform",
    param_names: list[str] | None = None,
) -> dict:
    """Fit `model(t, *params)` to (t, y) via nonlinear least squares.

    Returns dict with param estimates, standard errors, R^2, AIC, AICc, and
    the predicted/residual arrays for plotting.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    w = _weights(y, weight_scheme)
    sigma = 1.0 / np.sqrt(w)

    kwargs = {}
    if bounds is not None:
        kwargs["bounds"] = bounds

    popt, pcov = curve_fit(model, t, y, p0=p0, sigma=sigma, absolute_sigma=False, maxfev=10000, **kwargs)

    y_pred = model(t, *popt)
    residuals = y - y_pred

    ss_res = float(np.sum(w * residuals ** 2))
    ss_tot = float(np.sum(w * (y - np.average(y, weights=w)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    n = len(y)
    k = len(popt)
    # Gaussian-likelihood AIC using weighted RSS as proxy for -2*logL (standard
    # practice for weighted nonlinear regression AIC comparison).
    aic = n * np.log(ss_res / n) + 2 * k if ss_res > 0 else float("-inf")
    aicc = aic + (2 * k * (k + 1)) / (n - k - 1) if n - k - 1 > 0 else float("nan")

    se = np.sqrt(np.diag(pcov)) if pcov is not None else np.full(k, np.nan)
    cv_pct = 100.0 * se / np.abs(popt)

    names = param_names or [f"p{i}" for i in range(k)]
    return {
        "params": dict(zip(names, popt.tolist())),
        "param_se": dict(zip(names, se.tolist())),
        "param_cv_pct": dict(zip(names, cv_pct.tolist())),
        "r_squared": float(r_squared),
        "aic": float(aic),
        "aicc": float(aicc),
        "residuals": residuals,
        "predicted": y_pred,
        "covariance": pcov,
    }
