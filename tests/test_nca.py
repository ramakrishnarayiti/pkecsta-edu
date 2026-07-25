"""NCA validation: analytical round-trip + a textbook worked example.

Analytical case: pure mono-exponential decay C(t) = C0 * exp(-k*t) (this is
exactly the IV bolus 1-compartment concentration profile). For this curve:
  - AUC(0-inf) = C0 / k                       (closed form)
  - linear-up/log-down trapezoidal AUC(0-t) is EXACT between any two sample
    points on a pure exponential (the log trapezoid formula is derived from
    the exponential itself), so it should match closed form to float
    precision once extrapolated to infinity.
  - lambda_z recovered by the log-linear regression should equal k exactly
    (noise-free points on a line in log-space).
"""
import numpy as np
import pytest

from pkpd.pk.nca import (
    auc_linear,
    auc_linear_interp,
    auc_linear_log,
    auc_linear_up_log_down,
    bolus_c0,
    compute_flags,
    format_core_output,
    lambda_z,
    nca_iv_bolus,
    steady_state_metrics,
)

C0 = 100.0
K = 0.15  # 1/hr


def exponential_curve(n=8, t_end=24.0):
    t = np.linspace(0, t_end, n)
    c = C0 * np.exp(-K * t)
    return t, c


def test_lambda_z_recovers_true_k_exactly():
    t, c = exponential_curve()
    result = lambda_z(t, c)
    assert result["lambda_z"] == pytest.approx(K, rel=1e-9)
    assert result["half_life"] == pytest.approx(np.log(2) / K, rel=1e-9)


def test_log_down_auc_matches_analytical_auc_inf():
    t, c = exponential_curve()
    auc_t = auc_linear_up_log_down(t, c)
    lz = lambda_z(t, c)
    auc_inf = auc_t + c[-1] / lz["lambda_z"]
    analytical_auc_inf = C0 / K
    assert auc_inf == pytest.approx(analytical_auc_inf, rel=1e-9)


def test_linear_trapezoid_overestimates_vs_log_down_for_decay():
    t, c = exponential_curve(n=4)  # sparse sampling exaggerates the gap
    linear = auc_linear(t, c)
    log_down = auc_linear_up_log_down(t, c)
    assert linear > log_down


def test_nca_iv_bolus_full_profile_matches_closed_form():
    dose = 500.0
    t, c = exponential_curve()
    result = nca_iv_bolus(t, c, dose=dose, auc_method="linear_up_log_down")

    assert result["cmax"] == pytest.approx(C0)
    assert result["tmax"] == pytest.approx(0.0)
    assert result["lambda_z"] == pytest.approx(K, rel=1e-9)
    assert result["auc_inf"] == pytest.approx(dose / (dose / C0 * K) if False else C0 / K, rel=1e-9)

    # CL = Dose / AUC_inf ; for 1-compartment IV bolus, Vz = Dose / (C0 * ... );
    # cross-check CL/Vz relation instead of re-deriving Vz independently:
    expected_cl = dose / (C0 / K)
    assert result["cl"] == pytest.approx(expected_cl, rel=1e-9)
    assert result["vz"] == pytest.approx(expected_cl / K, rel=1e-9)


def test_textbook_worked_example_gibaldi_style():
    # Simple hand-checkable dataset: linear trapezoidal AUC of a triangle-ish
    # profile, computed by hand: t=[0,1,2,4], c=[0,10,8,4]
    # segment AUCs (linear trapezoid): (0+10)/2*1=5, (10+8)/2*1=9, (8+4)/2*2=12
    # total = 26
    t = np.array([0.0, 1.0, 2.0, 4.0])
    c = np.array([0.0, 10.0, 8.0, 4.0])
    assert auc_linear(t, c) == pytest.approx(26.0)


def test_lambda_z_returns_none_for_insufficient_points():
    t = np.array([0.0, 1.0])
    c = np.array([10.0, 5.0])
    result = lambda_z(t, c)
    assert result["lambda_z"] is None


def test_lambda_z_returns_none_when_concentration_not_declining():
    t = np.linspace(0, 5, 6)
    c = np.full(6, 10.0)  # flat, slope ~0 -> not negative -> None
    result = lambda_z(t, c)
    assert result["lambda_z"] is None


# ---- new AUC methods ----

def test_linear_log_matches_hand_computed_value_with_rising_segment():
    # t=[0,1,2,4], c=[0,5,10,4]: segment (0,1) falls back to linear since
    # c0=0; segment (1,2) is rising (5->10) and both linear_log and
    # linear_up_log_down disagree here (log-trap vs linear-trap); segment
    # (2,4) declines and both agree (log-trap).
    t = np.array([0.0, 1.0, 2.0, 4.0])
    c = np.array([0.0, 5.0, 10.0, 4.0])
    assert auc_linear_log(t, c) == pytest.approx(22.809755219692313)
    assert auc_linear_up_log_down(t, c) == pytest.approx(23.096280015247498)


def test_linear_log_and_interp_fallback_to_linear_on_zero_concentration():
    t = np.array([0.0, 1.0, 2.0, 3.0])
    c = np.array([5.0, 0.0, 5.0, 2.0])
    manual_linear = sum((c[i] + c[i + 1]) / 2 * (t[i + 1] - t[i]) for i in range(len(t) - 1))
    assert auc_linear_interp(t, c) == pytest.approx(manual_linear)
    # linear_log still log-traps the positive declining segment (2,3), so it
    # doesn't equal plain linear overall — just must not raise on the zero.
    assert auc_linear_log(t, c) == pytest.approx(8.274070003811875)


def test_linear_interp_matches_linear_on_full_profile():
    t = np.array([0.0, 2.0, 4.0])
    c = np.array([10.0, 20.0, 5.0])
    assert auc_linear_interp(t, c) == pytest.approx(np.trapezoid(c, t))


def test_linear_interp_partial_area_uses_log_interpolated_bounds():
    t = np.array([0.0, 2.0, 4.0])
    c = np.array([10.0, 20.0, 5.0])
    # C(1) log-interpolated between (0,10) and (2,20); C(3) log-interpolated
    # between (2,20) and (4,5); partial AUC sums linear trapezoids over
    # [1, 2, 3] with those interpolated endpoints.
    assert auc_linear_interp(t, c, 1.0, 3.0) == pytest.approx(32.071067811865476)


# ---- lambda_z Time Range mode ----

def test_lambda_z_time_range_matches_best_fit_when_range_equals_auto_window():
    t, c = exponential_curve()
    auto = lambda_z(t, c)
    lo, hi = min(auto["terminal_t"]), max(auto["terminal_t"])
    ranged = lambda_z(t, c, t_range=(lo, hi))
    assert ranged["lambda_z"] == pytest.approx(auto["lambda_z"], rel=1e-9)
    assert ranged["n_points"] == auto["n_points"]


def test_lambda_z_time_range_returns_none_when_too_few_points_in_range():
    t, c = exponential_curve()
    result = lambda_z(t, c, t_range=(0.0, 3.0))  # only first ~1 point in range
    assert result["lambda_z"] is None


# ---- lambda_z manual exclusion: must not affect AUC ----

def test_lambda_z_exclusion_changes_slope_but_not_auc():
    t, c = exponential_curve()
    c = c.copy()
    c[-2] *= 1.5  # perturb one terminal point off the log-linear line

    no_excl = nca_iv_bolus(t, c, dose=500.0, auc_method="linear_up_log_down")
    excl = nca_iv_bolus(t, c, dose=500.0, auc_method="linear_up_log_down",
                         lz_excluded_times={t[-2]})

    assert excl["auc_t"] == pytest.approx(no_excl["auc_t"])
    assert excl["lambda_z"] != pytest.approx(no_excl["lambda_z"])
    assert t[-2] not in excl["terminal_t"]


# ---- AUCINF(pred) / %Extrap ----

def test_aucinf_pred_matches_observed_on_noise_free_exponential():
    t, c = exponential_curve()
    result = nca_iv_bolus(t, c, dose=500.0, auc_method="linear_up_log_down")
    assert result["auc_inf_pred"] == pytest.approx(result["auc_inf"], rel=1e-9)


def test_pct_extrap_matches_hand_formula():
    t, c = exponential_curve()
    result = nca_iv_bolus(t, c, dose=500.0, auc_method="linear_up_log_down")
    expected = (result["auc_inf"] - result["auc_t"]) / result["auc_inf"] * 100.0
    assert result["pct_extrap"] == pytest.approx(expected)


# ---- bolus C0 back-extrapolation ----

def test_bolus_c0_recovers_true_c0_on_noise_free_exponential():
    # first sample at t=1, not t=0 -- the realistic bolus post-dose case
    t = np.array([1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
    c = C0 * np.exp(-K * t)
    assert bolus_c0(t, c) == pytest.approx(C0, rel=1e-9)


def test_bolus_c0_falls_back_to_first_point_when_rising():
    t = np.array([1.0, 2.0, 4.0])
    c = np.array([5.0, 8.0, 3.0])  # rising first segment: extrapolation undefined
    assert bolus_c0(t, c) == pytest.approx(5.0)


def test_bolus_c0_falls_back_to_first_point_with_single_observation():
    t = np.array([1.0])
    c = np.array([5.0])
    assert bolus_c0(t, c) == pytest.approx(5.0)


def test_bolus_c0_no_extrapolation_needed_when_first_sample_at_t0():
    t, c = exponential_curve()
    assert bolus_c0(t, c) == pytest.approx(c[0])


def test_nca_iv_bolus_inserts_extrapolated_c0_when_missing_t0():
    t = np.array([1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
    c = C0 * np.exp(-K * t)
    result = nca_iv_bolus(t, c, dose=500.0, auc_method="linear_up_log_down")
    # Cmax/Tmax now reflect the inserted t=0 point, not the first observation
    assert result["cmax"] == pytest.approx(C0, rel=1e-9)
    assert result["tmax"] == pytest.approx(0.0)
    assert result["pct_back_ext"] is not None
    assert result["pct_back_ext"] > 0


def test_nca_iv_bolus_no_back_extrap_pct_when_t0_already_present():
    t, c = exponential_curve()
    result = nca_iv_bolus(t, c, dose=500.0, auc_method="linear_up_log_down")
    assert result["pct_back_ext"] is None


# ---- steady state (Tau) ----

def test_steady_state_metrics_match_hand_computed_values():
    t = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 12.0])
    c = np.array([20.0, 45.0, 38.0, 25.0, 12.0, 8.0])
    tau = 12.0
    result = steady_state_metrics(t, c, tau, auc_method="linear", lambda_z_value=0.2)

    assert result["cmax_ss"] == pytest.approx(45.0)
    assert result["tmax_ss"] == pytest.approx(1.0)
    assert result["cmin_ss"] == pytest.approx(8.0)
    assert result["ctau"] == pytest.approx(8.0)  # tau lands exactly on the last observed point

    expected_auc_tau = np.trapezoid(c, t)
    assert result["auc_tau"] == pytest.approx(expected_auc_tau)
    expected_cavg = expected_auc_tau / tau
    assert result["cavg_ss"] == pytest.approx(expected_cavg)
    assert result["pct_fluctuation"] == pytest.approx((45.0 - 8.0) / expected_cavg * 100.0)
    assert result["accumulation_index"] == pytest.approx(1.0 / (1.0 - np.exp(-0.2 * tau)))


def test_steady_state_metrics_none_accumulation_without_lambda_z():
    t = np.array([0.0, 4.0, 8.0, 12.0])
    c = np.array([20.0, 25.0, 12.0, 8.0])
    result = steady_state_metrics(t, c, 12.0, auc_method="linear", lambda_z_value=None)
    assert result["accumulation_index"] is None


def test_nca_iv_bolus_with_tau_merges_steady_state_keys():
    t = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 12.0])
    c = np.array([20.0, 45.0, 38.0, 25.0, 12.0, 8.0])
    result = nca_iv_bolus(t, c, dose=500.0, auc_method="linear", tau=12.0)
    assert result["tau"] == pytest.approx(12.0)
    assert result["cmax_ss"] == pytest.approx(45.0)
    assert result["cavg_ss"] is not None


def test_nca_iv_bolus_without_tau_has_no_steady_state_keys():
    t, c = exponential_curve()
    result = nca_iv_bolus(t, c, dose=500.0, auc_method="linear_up_log_down")
    assert "cmax_ss" not in result


# ---- weight-normalized (mg/kg) params ----

def test_nca_iv_bolus_weight_normalization_matches_hand_formula():
    t, c = exponential_curve()
    dose = 500.0
    weight = 70.0
    result = nca_iv_bolus(t, c, dose=dose, auc_method="linear_up_log_down", weight=weight)
    assert result["dose_per_kg"] == pytest.approx(dose / weight)
    assert result["cl_per_kg"] == pytest.approx(result["cl"] / weight)
    assert result["vz_per_kg"] == pytest.approx(result["vz"] / weight)
    assert result["vss_per_kg"] == pytest.approx(result["vss"] / weight)


def test_nca_iv_bolus_no_weight_normalization_when_weight_not_given():
    t, c = exponential_curve()
    result = nca_iv_bolus(t, c, dose=500.0, auc_method="linear_up_log_down")
    assert result["dose_per_kg"] is None
    assert result["cl_per_kg"] is None


# ---- acceptance-criteria flags ----

def test_flag_n_samples_insufficient_below_min_samples():
    result = nca_iv_bolus(np.array([0.0, 1.0]), np.array([10.0, 5.0]), dose=500.0)
    assert result["flag_n_samples"] == "Insufficient"
    assert result["span"] is None
    assert result["flag_low_rsq"] is None


def test_flag_n_samples_ok_and_span_computed_for_valid_profile():
    t, c = exponential_curve()
    result = nca_iv_bolus(t, c, dose=500.0, auc_method="linear_up_log_down")
    assert result["flag_n_samples"] == "OK"
    half_life = np.log(2) / K
    expected_span = (max(result["terminal_t"]) - min(result["terminal_t"])) / half_life
    assert result["span"] == pytest.approx(expected_span)
    assert result["flag_low_rsq"] is False  # noise-free line, R^2 == 1


def test_compute_flags_flags_low_rsq_directly():
    lz = {"lambda_z": 0.1, "half_life": 6.93, "terminal_t": [0.0, 4.0], "r_squared": 0.5}
    flags = compute_flags(n_points=5, lz=lz, pct_extrap=5.0)
    assert flags["flag_low_rsq"] is True
    assert flags["flag_high_extrap"] is False


def test_compute_flags_flags_high_extrap_directly():
    lz = {"lambda_z": 0.1, "half_life": 6.93, "terminal_t": [0.0, 20.0], "r_squared": 0.95}
    flags = compute_flags(n_points=5, lz=lz, pct_extrap=25.0)
    assert flags["flag_high_extrap"] is True
    assert flags["flag_low_span"] is False  # span = 20/6.93 ~= 2.89 >= 2


# ---- Core Output export ----

def test_format_core_output_includes_settings_and_results():
    t, c = exponential_curve()
    result = nca_iv_bolus(t, c, dose=500.0, auc_method="linear_up_log_down")
    settings = {"subject": "1", "route": "iv_bolus", "dose": 500.0}
    text = format_core_output(result, settings)
    assert "subject: 1" in text
    assert "route: iv_bolus" in text
    assert f"lambda_z: {result['lambda_z']}" in text
    assert "cmax:" in text


def test_format_core_output_reports_no_warnings_for_clean_profile():
    lz = {"lambda_z": 0.1, "half_life": 6.93, "terminal_t": [0.0, 20.0], "r_squared": 0.99}
    flags = compute_flags(n_points=10, lz=lz, pct_extrap=5.0)
    result = {**flags}
    text = format_core_output(result, settings={})
    assert "Warnings:" in text
    assert "none" in text


def test_format_core_output_reports_warning_for_flagged_profile():
    t = np.array([0.0, 1.0])
    c = np.array([10.0, 5.0])
    result = nca_iv_bolus(t, c, dose=500.0)  # too few points -> Insufficient
    text = format_core_output(result, settings={})
    assert "Insufficient samples" in text
