"""Closed-form analytical concentration equations, standard textbook forms
(Gibaldi & Perrier / Rowland & Tozer). Used directly instead of ODE solving —
faster and numerically exact for linear PK.

2-compartment IV infusion and extravascular routes are not yet implemented
(the algebra is more error-prone to derive/verify than IV bolus; shipping
fewer well-tested models beats shipping more under-tested ones — see
PLANNING.md's "zero errors in math" constraint). 1-compartment covers all
three routes; 2-compartment covers IV bolus only for now.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-8


def conc_1c_iv_bolus(t: np.ndarray, k: float, V: float, dose: float) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return (dose / V) * np.exp(-k * t)


def conc_1c_iv_bolus_cl(t: np.ndarray, cl: float, V: float, dose: float) -> np.ndarray:
    """Clearance parameterization of the same 1-compartment IV bolus model
    (k = Cl/V) — PK01-style Cl/V fit, preferred over K/V for its lower
    parameter correlation (Cl and V estimate near-independently)."""
    t = np.asarray(t, dtype=float)
    return (dose / V) * np.exp(-(cl / V) * t)


def conc_1c_iv_infusion(t: np.ndarray, k: float, V: float, dose: float, tinf: float) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    k0 = dose / tinf
    during = (k0 / (k * V)) * (1.0 - np.exp(-k * t))
    c_end = (k0 / (k * V)) * (1.0 - np.exp(-k * tinf))
    after = c_end * np.exp(-k * (t - tinf))
    return np.where(t <= tinf, during, after)


def conc_1c_extravascular(t: np.ndarray, ka: float, k: float, V: float, dose: float, tlag: float = 0.0) -> np.ndarray:
    """V here is apparent volume V/F (F not separately identifiable from
    plasma data alone without an IV arm)."""
    t = np.asarray(t, dtype=float)
    tt = t - tlag
    if abs(ka - k) < _EPS:
        # flip-flop limiting case (ka -> k): C(t) = Dose*k/V * t * exp(-k*t)
        c = (dose * k / V) * tt * np.exp(-k * tt)
    else:
        c = (dose * ka) / (V * (ka - k)) * (np.exp(-k * tt) - np.exp(-ka * tt))
    return np.where(tt < 0, 0.0, c)


def conc_2c_iv_bolus(t: np.ndarray, A: float, alpha: float, B: float, beta: float) -> np.ndarray:
    """Hybrid biexponential form: C(t) = A*exp(-alpha*t) + B*exp(-beta*t).
    A, B carry the dose/V1 scaling already (standard WinNonlin-style
    polyexponential parameterization); convert to micro-constants
    (k10, k12, k21, V1) with `micro_constants_from_hybrid` if needed."""
    t = np.asarray(t, dtype=float)
    return A * np.exp(-alpha * t) + B * np.exp(-beta * t)


def micro_constants_from_hybrid(A: float, alpha: float, B: float, beta: float, dose: float) -> dict:
    """Convert fitted hybrid (A, alpha, B, beta) to compartmental
    micro-constants (Gibaldi & Perrier eq. for 2-compartment IV bolus)."""
    V1 = dose / (A + B)
    k21 = (A * beta + B * alpha) / (A + B)
    k10 = (alpha * beta) / k21
    k12 = alpha + beta - k10 - k21
    return {"V1": V1, "k10": k10, "k12": k12, "k21": k21}
