"""Analytical round-trip: generate synthetic data from known true params
using the closed-form equation, fit it back, assert recovered params match
true values (noise-free -> tight tolerance)."""
import numpy as np
import pytest
from functools import partial

from pkpd.pk.compartmental.models import (
    conc_1c_extravascular,
    conc_1c_iv_bolus,
    conc_1c_iv_bolus_cl,
    conc_1c_iv_infusion,
    conc_2c_iv_bolus,
    micro_constants_from_hybrid,
)
from pkpd.pk.compartmental.fitting import fit_model


def test_1c_iv_bolus_recovers_true_params():
    true_k, true_V, dose = 0.2, 10.0, 100.0
    t = np.linspace(0.1, 24, 12)
    y = conc_1c_iv_bolus(t, true_k, true_V, dose)

    model = partial(conc_1c_iv_bolus, dose=dose)
    result = fit_model(model, t, y, p0=[0.1, 5.0], bounds=([1e-6, 1e-6], [10, 100]),
                        param_names=["k", "V"])

    assert result["params"]["k"] == pytest.approx(true_k, rel=1e-4)
    assert result["params"]["V"] == pytest.approx(true_V, rel=1e-4)
    assert result["r_squared"] == pytest.approx(1.0, abs=1e-6)


def test_1c_iv_bolus_cl_recovers_true_params():
    true_cl, true_V, dose = 2.0, 10.0, 100.0
    t = np.linspace(0.1, 24, 12)
    y = conc_1c_iv_bolus_cl(t, true_cl, true_V, dose)

    model = partial(conc_1c_iv_bolus_cl, dose=dose)
    result = fit_model(model, t, y, p0=[1.0, 5.0], bounds=([1e-6, 1e-6], [10, 100]),
                        param_names=["Cl", "V"])

    assert result["params"]["Cl"] == pytest.approx(true_cl, rel=1e-4)
    assert result["params"]["V"] == pytest.approx(true_V, rel=1e-4)
    assert result["r_squared"] == pytest.approx(1.0, abs=1e-6)
    assert np.isfinite(result["param_cv_pct"]["Cl"])
    assert np.isfinite(result["param_cv_pct"]["V"])


def test_1c_iv_infusion_recovers_true_params():
    true_k, true_V, dose, tinf = 0.15, 20.0, 500.0, 2.0
    t = np.array([0.5, 1.0, 1.5, 2.0, 3, 5, 8, 12, 18, 24])
    y = conc_1c_iv_infusion(t, true_k, true_V, dose, tinf)

    model = partial(conc_1c_iv_infusion, dose=dose, tinf=tinf)
    result = fit_model(model, t, y, p0=[0.1, 10.0], bounds=([1e-6, 1e-6], [10, 200]),
                        param_names=["k", "V"])

    assert result["params"]["k"] == pytest.approx(true_k, rel=1e-4)
    assert result["params"]["V"] == pytest.approx(true_V, rel=1e-4)


def test_1c_extravascular_recovers_true_params():
    true_ka, true_k, true_V, dose = 1.2, 0.15, 15.0, 200.0
    t = np.array([0.25, 0.5, 1, 2, 4, 6, 8, 12, 18, 24])
    y = conc_1c_extravascular(t, true_ka, true_k, true_V, dose)

    model = partial(conc_1c_extravascular, dose=dose)
    result = fit_model(model, t, y, p0=[0.5, 0.1, 10.0],
                        bounds=([1e-6, 1e-6, 1e-6], [20, 10, 200]),
                        param_names=["ka", "k", "V"])

    assert result["params"]["ka"] == pytest.approx(true_ka, rel=1e-3)
    assert result["params"]["k"] == pytest.approx(true_k, rel=1e-3)
    assert result["params"]["V"] == pytest.approx(true_V, rel=1e-3)


def test_1c_extravascular_flip_flop_limit_matches_general_form_nearby():
    # As ka -> k, the general closed form is numerically unstable (0/0);
    # confirm the limiting-case branch agrees with the general formula
    # evaluated just off the singularity.
    k = 0.2
    dose, V = 100.0, 10.0
    t = np.linspace(0.1, 20, 15)

    c_limit = conc_1c_extravascular(t, ka=k, k=k, V=V, dose=dose)  # exact ka==k
    c_near = conc_1c_extravascular(t, ka=k + 1e-5, k=k, V=V, dose=dose)  # just off

    assert np.allclose(c_limit, c_near, rtol=1e-3)


def test_2c_iv_bolus_recovers_true_hybrid_params_and_micro_constants():
    dose = 1000.0
    true_A, true_alpha, true_B, true_beta = 8.0, 1.5, 2.0, 0.1
    t = np.array([0.1, 0.25, 0.5, 1, 2, 4, 8, 12, 18, 24])
    y = conc_2c_iv_bolus(t, true_A, true_alpha, true_B, true_beta)

    result = fit_model(
        conc_2c_iv_bolus, t, y, p0=[5.0, 1.0, 1.0, 0.2],
        bounds=([1e-6, 1e-6, 1e-6, 1e-6], [50, 10, 50, 5]),
        param_names=["A", "alpha", "B", "beta"],
    )

    p = result["params"]
    assert p["A"] == pytest.approx(true_A, rel=1e-3)
    assert p["alpha"] == pytest.approx(true_alpha, rel=1e-3)
    assert p["B"] == pytest.approx(true_B, rel=1e-3)
    assert p["beta"] == pytest.approx(true_beta, rel=1e-3)

    micro = micro_constants_from_hybrid(p["A"], p["alpha"], p["B"], p["beta"], dose)
    # sanity: microconstants must be positive (physically valid PK system)
    assert micro["V1"] > 0
    assert micro["k10"] > 0
    assert micro["k12"] > 0
    assert micro["k21"] > 0

    # round-trip check: regenerate concentration from recovered micro constants
    # via the alpha/beta relations and confirm they reproduce alpha, beta
    k10, k12, k21 = micro["k10"], micro["k12"], micro["k21"]
    sum_k = k10 + k12 + k21
    disc = np.sqrt(sum_k ** 2 - 4 * k10 * k21)
    alpha_check = (sum_k + disc) / 2
    beta_check = (sum_k - disc) / 2
    assert alpha_check == pytest.approx(true_alpha, rel=1e-3)
    assert beta_check == pytest.approx(true_beta, rel=1e-3)
