"""The golden matrix definition: fixed datasets, the case cross-product, and
the parameters locked against regression. Shared by `test_golden.py` (which
asserts) and `regen_golden.py` (which rewrites the committed values), so the
two can never drift apart.

Datasets are literal arrays, not generated ones — a golden master built on
`linspace` or a seeded RNG silently changes meaning the day a library
changes its output, which defeats the purpose.
"""
from __future__ import annotations

import numpy as np

from pkpd.pk import nca
from pkpd.pk.nca import AUC_METHODS

# Each profile deliberately starts after t=0 so the route-specific t=0
# insertion rule (back-extrapolated C0 / Cmin / zero) is exercised, and each
# has a clean enough tail for lambda_z to be estimable.
PROFILES = {
    "iv_bolus": {
        "time": [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0],
        "conc": [92.3, 85.1, 72.4, 52.8, 27.9, 7.8, 2.15, 0.16],
        "dose": 500.0,
        "infusion_duration": 0.0,
    },
    "iv_infusion": {
        "time": [0.5, 1.0, 2.0, 3.0, 4.0, 8.0, 12.0, 24.0],
        "conc": [12.4, 23.1, 41.6, 38.2, 30.5, 15.1, 7.4, 0.92],
        "dose": 500.0,
        "infusion_duration": 2.0,
    },
    "extravascular": {
        "time": [0.5, 1.0, 2.0, 3.0, 4.0, 8.0, 12.0, 24.0],
        "conc": [8.2, 19.6, 34.1, 39.8, 37.2, 21.3, 11.6, 1.94],
        "dose": 500.0,
        "infusion_duration": 0.0,
    },
}

TAU = 12.0
WEIGHT = 70.0

# Everything downstream of the AUC rule and the lambda_z fit. If a rule
# changes for one route/method pair, at least one of these moves.
LOCKED_PARAMS = [
    "cmax", "tmax", "clast", "tlast",
    "auc_t", "auc_inf", "auc_inf_pred", "pct_extrap", "pct_back_ext",
    "aumc_t", "aumc_inf", "mrt",
    "lambda_z", "half_life", "n_points", "r_squared", "adj_r_squared",
    "cl", "vz", "vss", "cl_per_kg", "vz_per_kg", "vss_per_kg", "dose_per_kg",
    "span", "flag_n_samples", "flag_low_rsq", "flag_high_extrap", "flag_low_span",
    "n_excluded",
    # steady-state block, present only on the tau cases
    "cmax_ss", "tmax_ss", "cmin_ss", "ctau", "auc_tau", "cavg_ss",
    "pct_fluctuation", "accumulation_index", "flag_tau_beyond_tlast",
]

_ROUTE_FUNCTIONS = {
    "iv_bolus": nca.nca_iv_bolus,
    "iv_infusion": nca.nca_iv_infusion,
    "extravascular": nca.nca_extravascular,
}


def _build_cases() -> dict[str, dict]:
    cases = {}
    for route in PROFILES:
        for method in AUC_METHODS:
            for mode in ("single", "steady_state"):
                name = f"{route}__{method}__{mode}"
                cases[name] = {
                    "route": route,
                    "auc_method": method,
                    "tau": TAU if mode == "steady_state" else None,
                }
    return cases


CASES = _build_cases()


def run_case(case_name: str) -> dict:
    """Run one matrix cell and return its results as plain JSON-able types."""
    case = CASES[case_name]
    profile = PROFILES[case["route"]]

    kwargs = {
        "auc_method": case["auc_method"],
        "tau": case["tau"],
        "weight": WEIGHT,
    }
    if case["route"] == "iv_infusion":
        kwargs["infusion_duration"] = profile["infusion_duration"]

    result = _ROUTE_FUNCTIONS[case["route"]](
        np.array(profile["time"], dtype=float),
        np.array(profile["conc"], dtype=float),
        dose=profile["dose"],
        **kwargs,
    )
    return {key: _plain(result.get(key)) for key in LOCKED_PARAMS if key in result}


def _plain(value):
    """NumPy scalars and bools don't serialize; results must round-trip
    through JSON unchanged or the comparison is against the wrong thing."""
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value
