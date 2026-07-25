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


_LOG_FLOOR = 1e-12


def fit_model(
    model: Callable,
    t: np.ndarray,
    y: np.ndarray,
    p0: list[float],
    bounds: tuple[list[float], list[float]] | None = None,
    weight_scheme: WeightScheme = "uniform",
    param_names: list[str] | None = None,
    log_residuals: bool = False,
) -> dict:
    """Fit `model(t, *params)` to (t, y) via nonlinear least squares.

    log_residuals fits on log-concentration instead of concentration, which
    stabilizes variance across the orders of magnitude a PK curve spans —
    without it, the two or three highest early points dominate the objective
    and the terminal phase is fitted badly. Non-positive observations carry
    no information on a log scale and are dropped from the fit (reported as
    `n_excluded`) rather than clipped, which would plant a large artificial
    residual at whatever floor was chosen.

    Returns dict with param estimates, standard errors, R^2, AIC, AICc, and
    the predicted/residual arrays for plotting. Predicted values and
    residuals always come back on the original concentration scale (that is
    what a residual plot needs); R^2 and AIC are computed on whichever scale
    was actually fitted, since that is the objective that was minimized.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    n_excluded = 0
    if log_residuals:
        usable = y > 0
        n_excluded = int((~usable).sum())
        t_fit, y_obs = t[usable], np.log(y[usable])

        def fit_target(tt, *params):
            return np.log(np.clip(model(tt, *params), _LOG_FLOOR, None))
    else:
        t_fit, y_obs = t, y
        fit_target = model

    # 1/Y weighting on a log scale is meaningless — log concentrations go
    # negative, and 1/(a small negative) is a huge weight on an arbitrary
    # point. The log transform IS the variance stabilizer, so it replaces
    # the weighting scheme rather than stacking with it.
    w = np.ones_like(y_obs) if log_residuals else _weights(y_obs, weight_scheme)
    sigma = 1.0 / np.sqrt(w)

    kwargs = {}
    if bounds is not None:
        kwargs["bounds"] = bounds

    popt, pcov = curve_fit(fit_target, t_fit, y_obs, p0=p0, sigma=sigma,
                            absolute_sigma=False, maxfev=10000, **kwargs)

    # Diagnostics on the fitted scale; plotting arrays on the original one.
    y_obs_pred = fit_target(t_fit, *popt)
    ss_res = float(np.sum(w * (y_obs - y_obs_pred) ** 2))
    ss_tot = float(np.sum(w * (y_obs - np.average(y_obs, weights=w)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    y_pred = model(t, *popt)
    residuals = y - y_pred

    n = len(y_obs)
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
        # AIC/R^2 are only comparable between fits made on the same scale —
        # a log-scale AIC must never be ranked against a linear-scale one.
        "fit_scale": "log" if log_residuals else "linear",
        "n_excluded": n_excluded,
        "residuals": residuals,
        "predicted": y_pred,
        "covariance": pcov,
    }
