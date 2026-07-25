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
    aumc_linear,
    aumc_linear_up_log_down,
    bolus_c0,
    compute_flags,
    format_core_output,
    lambda_z,
    nca_extravascular,
    nca_iv_bolus,
    nca_iv_infusion,
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
    lz = {"lambda_z": 0.1, "half_life": 6.93, "terminal_t": [0.0, 4.0], "r_squared": 0.5, "adj_r_squared": 0.5}
    flags = compute_flags(n_points=5, lz=lz, pct_extrap=5.0)
    assert flags["flag_low_rsq"] is True
    assert flags["flag_high_extrap"] is False


def test_compute_flags_flags_high_extrap_directly():
    lz = {"lambda_z": 0.1, "half_life": 6.93, "terminal_t": [0.0, 20.0], "r_squared": 0.95, "adj_r_squared": 0.95}
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
    lz = {"lambda_z": 0.1, "half_life": 6.93, "terminal_t": [0.0, 20.0], "r_squared": 0.99, "adj_r_squared": 0.99}
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


# ---- AUMC follows the chosen AUC method ----

def test_log_down_aumc_matches_analytical_aumc_inf():
    # For C(t) = C0*exp(-k*t): AUMC(0-inf) = C0/k^2, and the log-moment
    # trapezoid is exact between any two points on that curve — so the
    # extrapolated total must hit the closed form to float precision.
    t, c = exponential_curve()
    aumc_t = aumc_linear_up_log_down(t, c)
    clast, tlast = c[-1], t[-1]
    aumc_inf = aumc_t + (clast * tlast) / K + clast / K ** 2
    assert aumc_inf == pytest.approx(C0 / K ** 2, rel=1e-9)


def test_linear_and_log_down_aumc_differ_on_a_declining_profile():
    # Direction isn't fixed (t*C is not monotone), but the two rules must
    # not silently collapse into the same number — that was the bug: AUMC
    # was always linear regardless of the AUC method chosen.
    t, c = exponential_curve(n=4)
    assert aumc_linear(t, c) != pytest.approx(aumc_linear_up_log_down(t, c))


def test_nca_uses_matching_aumc_for_the_selected_auc_method():
    t, c = exponential_curve()
    log_down = nca_iv_bolus(t, c, dose=500.0, auc_method="linear_up_log_down")
    linear = nca_iv_bolus(t, c, dose=500.0, auc_method="linear")
    assert log_down["aumc_t"] == pytest.approx(aumc_linear_up_log_down(t, c))
    assert linear["aumc_t"] == pytest.approx(aumc_linear(t, c))
    # MRT for a 1-compartment bolus is exactly 1/k; only the consistent
    # log-down pairing recovers it.
    assert log_down["mrt"] == pytest.approx(1.0 / K, rel=1e-9)


# ---- Best Fit excludes the absorption limb for non-bolus routes ----

def test_best_fit_lambda_z_ignores_pre_tmax_points_extravascular():
    # Absorption limb then clean log-linear decay. Including the rising
    # points would drag lambda_z away from the true terminal K.
    t = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
    c = np.array([0.0, 30.0, 50.0, 60.0, *(60.0 * np.exp(-K * (t[4:] - 2.0)))])
    result = nca_extravascular(t, c, dose=500.0, auc_method="linear_up_log_down")
    assert result["lambda_z"] == pytest.approx(K, rel=1e-9)
    assert min(result["terminal_t"]) > result["tmax"]


def test_iv_bolus_best_fit_still_allows_the_first_point():
    t, c = exponential_curve()
    result = nca_iv_bolus(t, c, dose=500.0, auc_method="linear_up_log_down")
    assert result["n_points"] == len(t)  # whole curve is the terminal phase


def test_explicit_time_range_overrides_pre_tmax_exclusion():
    t = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 12.0])
    c = np.array([1.0, 40.0, 60.0, 30.0, 12.0, 5.0])
    result = nca_extravascular(t, c, dose=500.0, lz_t_range=(1.0, 12.0))
    assert min(result["terminal_t"]) == pytest.approx(1.0)


# ---- degenerate / missing data ----

def test_nan_concentrations_are_excluded_not_propagated():
    t, c = exponential_curve()
    c = c.copy()
    c[3] = np.nan  # a BQL/blank cell
    result = nca_iv_bolus(t, c, dose=500.0, auc_method="linear_up_log_down")
    assert result["n_excluded"] == 1
    assert np.isfinite(result["auc_t"])
    assert np.isfinite(result["cl"])
    assert "excluded as missing/BQL" in format_core_output(result, settings={})


def test_all_missing_profile_returns_a_result_instead_of_crashing():
    result = nca_iv_bolus(np.array([0.0, 1.0]), np.array([np.nan, np.nan]), dose=500.0)
    assert result["cmax"] is None
    assert result["flag_n_samples"] == "Insufficient"


def test_degenerate_result_has_the_same_keys_as_a_normal_one():
    """A degenerate profile must return the same shape as a good one, or
    callers have to guess which keys exist. This is what keeps _NULL_RESULT
    in step with the parameters _nca_common actually produces."""
    t, c = exponential_curve()
    normal = nca_iv_bolus(t, c, dose=500.0, weight=70.0)
    degenerate = nca_iv_bolus(np.array([0.0, 1.0]), np.array([np.nan, np.nan]), dose=500.0)

    # `note` only appears on the degenerate path; everything else must match.
    assert set(normal) | {"note"} == set(degenerate)
    for key in normal:
        if key not in ("route", "dose", "n_excluded", "flag_n_samples", "n_points"):
            assert degenerate[key] is None, f"{key} should be None on a degenerate profile"


def test_single_point_profile_still_reports_cmax():
    result = nca_iv_bolus(np.array([2.0]), np.array([10.0]), dose=500.0)
    assert result["cmax"] == pytest.approx(10.0)
    assert result["lambda_z"] is None


def test_unsorted_input_is_sorted_before_analysis():
    t, c = exponential_curve()
    order = np.array([3, 0, 5, 1, 7, 2, 6, 4])
    shuffled = nca_iv_bolus(t[order], c[order], dose=500.0, auc_method="linear_up_log_down")
    straight = nca_iv_bolus(t, c, dose=500.0, auc_method="linear_up_log_down")
    assert shuffled["auc_t"] == pytest.approx(straight["auc_t"])
    assert shuffled["lambda_z"] == pytest.approx(straight["lambda_z"])


def test_tau_past_last_sample_is_flagged_not_an_index_error():
    t = np.array([0.0, 1.0, 2.0, 4.0, 8.0])
    c = np.array([20.0, 45.0, 38.0, 25.0, 12.0])
    result = steady_state_metrics(t, c, tau=24.0, auc_method="linear", lambda_z_value=0.2)
    assert result["flag_tau_beyond_tlast"] is True
    assert result["ctau"] == pytest.approx(12.0)  # clamped to Clast, not extrapolated


def test_tau_within_profile_is_not_flagged():
    t = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 12.0])
    c = np.array([20.0, 45.0, 38.0, 25.0, 12.0, 8.0])
    result = steady_state_metrics(t, c, tau=12.0, auc_method="linear", lambda_z_value=0.2)
    assert result["flag_tau_beyond_tlast"] is False


# ---- t=0 insertion rules (Guide p.134) ----

def test_extravascular_inserts_zero_at_t0_when_first_sample_is_later():
    # No drug in plasma before an oral dose, so the (0, t1) triangle is part
    # of AUC. Omitting it understates early AUC for every profile not
    # sampled at t=0.
    t = np.array([1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
    c = np.array([30.0, 50.0, 60.0, 30.0, 12.0, 5.0])
    result = nca_extravascular(t, c, dose=500.0, auc_method="linear")

    missing_triangle = (0.0 + 30.0) / 2 * 1.0
    assert result["auc_t"] == pytest.approx(np.trapezoid(c, t) + missing_triangle)
    assert result["cmax"] == pytest.approx(60.0)  # inserted zero can't be Cmax


def test_extravascular_leaves_profile_alone_when_it_already_starts_at_t0():
    t = np.array([0.0, 1.0, 2.0, 4.0, 8.0])
    c = np.array([0.0, 30.0, 50.0, 30.0, 12.0])
    result = nca_extravascular(t, c, dose=500.0, auc_method="linear")
    assert result["auc_t"] == pytest.approx(np.trapezoid(c, t))


def test_infusion_also_starts_from_zero_at_t0():
    t = np.array([1.0, 2.0, 4.0, 8.0])
    c = np.array([20.0, 30.0, 18.0, 6.0])
    result = nca_iv_infusion(t, c, dose=500.0, infusion_duration=2.0, auc_method="linear")
    assert result["auc_t"] == pytest.approx(np.trapezoid(c, t) + (0.0 + 20.0) / 2 * 1.0)


def test_steady_state_inserts_cmin_at_t0_not_zero():
    # At steady state the interval starts from the trough, not from zero.
    t = np.array([1.0, 2.0, 4.0, 8.0, 12.0])
    c = np.array([30.0, 50.0, 38.0, 20.0, 15.0])
    result = nca_extravascular(t, c, dose=500.0, auc_method="linear", tau=12.0)

    cmin = 15.0
    assert result["auc_t"] == pytest.approx(np.trapezoid(c, t) + (cmin + 30.0) / 2 * 1.0)
    assert result["cmin_ss"] == pytest.approx(cmin)


def test_steady_state_cmin_comes_from_the_interval_not_the_whole_record():
    # Sampling runs to 24h but the dosing interval is 12h. The trough at t=24
    # belongs to a later interval and must not seed t=0 or become Cmin_ss.
    t = np.array([1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
    c = np.array([30.0, 50.0, 38.0, 20.0, 16.0, 2.0])
    result = nca_extravascular(t, c, dose=500.0, auc_method="linear", tau=12.0)
    assert result["cmin_ss"] == pytest.approx(16.0)  # not 2.0


def test_bolus_t0_insertion_still_uses_back_extrapolation_not_zero():
    t = np.array([1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
    c = C0 * np.exp(-K * t)
    result = nca_iv_bolus(t, c, dose=500.0, auc_method="linear_up_log_down")
    assert result["cmax"] == pytest.approx(C0, rel=1e-9)  # C0 > first sample


# ---- lambda_z explicit n_points floor ----

def test_lambda_z_rejects_explicit_n_points_below_the_minimum():
    t, c = exponential_curve()
    assert lambda_z(t, c, n_points=2)["lambda_z"] is None
    assert lambda_z(t, c, n_points=3)["lambda_z"] == pytest.approx(K, rel=1e-9)


# ---- infusion MRT correction ----

def test_infusion_mrt_subtracts_half_the_infusion_duration():
    t, c = exponential_curve()
    tinf = 2.0
    bolus = nca_iv_bolus(t, c, dose=500.0, auc_method="linear_up_log_down")
    infusion = nca_iv_infusion(t, c, dose=500.0, infusion_duration=tinf,
                                auc_method="linear_up_log_down")
    # same data, so the only MRT difference is the Tinf/2 correction
    assert infusion["mrt"] == pytest.approx(bolus["mrt"] - tinf / 2.0, rel=1e-9)
