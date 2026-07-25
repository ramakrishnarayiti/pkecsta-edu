"""Non-compartmental analysis. Pure functions: NumPy arrays in, dict out.
No Qt, no pandas dependency beyond the array extraction the caller does."""
from __future__ import annotations

import numpy as np

MIN_TERMINAL_POINTS = 3

# Acceptance-criteria thresholds (Guide p.130) — standard WinNonlin-style
# defaults. A profile failing these still reports its numbers; the flag is
# advisory, never a silent drop.
RSQ_ADJ_THRESHOLD = 0.80
PCT_EXTRAP_THRESHOLD = 20.0
SPAN_THRESHOLD = 2.0
MIN_SAMPLES = 3


def cmax_tmax(time: np.ndarray, conc: np.ndarray) -> dict:
    """Observed Cmax/Tmax, no interpolation."""
    idx = int(np.argmax(conc))
    return {"cmax": float(conc[idx]), "tmax": float(time[idx])}


def bolus_c0(time: np.ndarray, conc: np.ndarray) -> float:
    """Back-extrapolated C0 for IV bolus dosing: log-linear regression on
    the first two points, extrapolated to t=0. Falls back to the first
    observed concentration when there's no second point, either point is
    non-positive, concentration isn't declining (extrapolation undefined),
    or the extrapolated value is nonsensically below the first observation."""
    if time[0] == 0:
        return float(conc[0])
    if len(time) < 2 or conc[0] <= 0 or conc[1] <= 0 or conc[1] >= conc[0]:
        return float(conc[0])
    slope, intercept = np.polyfit([time[0], time[1]], [np.log(conc[0]), np.log(conc[1])], 1)
    c0 = float(np.exp(intercept))
    return c0 if c0 > conc[0] else float(conc[0])


def auc_linear(time: np.ndarray, conc: np.ndarray) -> float:
    """Linear trapezoidal AUC(0-t)."""
    return float(np.trapezoid(conc, time))


def auc_linear_up_log_down(time: np.ndarray, conc: np.ndarray) -> float:
    """Linear-up/log-down trapezoidal: linear trapezoid while concentration
    is rising or flat, log trapezoid while declining (avoids overestimating
    AUC during the log-linear elimination phase)."""
    auc = 0.0
    for i in range(len(time) - 1):
        t0, t1 = time[i], time[i + 1]
        c0, c1 = conc[i], conc[i + 1]
        dt = t1 - t0
        if c1 >= c0 or c0 <= 0 or c1 <= 0:
            auc += (c0 + c1) * dt / 2.0
        else:
            auc += (c0 - c1) * dt / np.log(c0 / c1)
    return float(auc)


def _interp_conc(t0: float, c0: float, t1: float, c1: float, t: float) -> float:
    """Concentration at t between two known points: log interpolation if
    both concentrations are positive and unequal, else linear."""
    if c0 <= 0 or c1 <= 0 or c0 == c1:
        return c0 + (c1 - c0) * (t - t0) / (t1 - t0)
    return c0 * np.exp((t - t0) / (t1 - t0) * np.log(c1 / c0))


def _point_conc(time: np.ndarray, conc: np.ndarray, t: float) -> float:
    """Concentration at t: exact if t is an observed point, else
    interpolated between the bracketing pair. Outside the observed range it
    clamps to the nearest endpoint — never extrapolates, never indexes off
    the end of the array (a Tau past the last sample used to IndexError)."""
    if t <= time[0]:
        return float(conc[0])
    if t >= time[-1]:
        return float(conc[-1])
    idx = int(np.searchsorted(time, t))
    if time[idx] == t:
        return float(conc[idx])
    return _interp_conc(time[idx - 1], conc[idx - 1], time[idx], conc[idx], t)


def auc_linear_log(time: np.ndarray, conc: np.ndarray) -> float:
    """Linear-log trapezoidal: log trapezoid on every segment where both
    concentrations are positive and unequal (rising or falling), linear
    trapezoid otherwise — unlike linear-up/log-down, log applies on rising
    segments too."""
    auc = 0.0
    for i in range(len(time) - 1):
        t0, t1 = time[i], time[i + 1]
        c0, c1 = conc[i], conc[i + 1]
        dt = t1 - t0
        if c0 <= 0 or c1 <= 0 or c0 == c1:
            auc += (c0 + c1) * dt / 2.0
        else:
            auc += (c0 - c1) * dt / np.log(c0 / c1)
    return float(auc)


def _clip_points(time: np.ndarray, conc: np.ndarray,
                  t_start: float | None, t_end: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Point set covering [t_start, t_end] (clipped to the observed range),
    with concentration at the two bounds interpolated (lin/log) unless they
    land exactly on an observed time. Shared by partial-area AUC and any
    AUC-over-a-window calculation (e.g. steady-state AUCtau)."""
    t_start = time[0] if t_start is None else max(t_start, time[0])
    t_end = time[-1] if t_end is None else min(t_end, time[-1])
    if t_end <= t_start:
        return np.array([]), np.array([])

    pts_t = [t_start]
    pts_c = [_point_conc(time, conc, t_start)]
    for i in range(len(time)):
        if t_start < time[i] < t_end:
            pts_t.append(time[i])
            pts_c.append(conc[i])
    pts_t.append(t_end)
    pts_c.append(_point_conc(time, conc, t_end))
    return np.array(pts_t), np.array(pts_c)


def auc_linear_interp(time: np.ndarray, conc: np.ndarray,
                       t_start: float | None = None, t_end: float | None = None) -> float:
    """Linear trapezoidal with linear/log interpolated endpoints. With no
    bounds, equivalent to auc_linear over the full profile. With bounds,
    interpolates concentration at t_start/t_end (clipped to the observed
    range) and sums linear trapezoids over the resulting point set — this
    is also the partial-area AUC calculation."""
    pts_t, pts_c = _clip_points(time, conc, t_start, t_end)
    if len(pts_t) == 0:
        return 0.0

    return float(np.trapezoid(pts_c, pts_t))


def aumc_linear(time: np.ndarray, conc: np.ndarray) -> float:
    """Linear trapezoidal AUMC(0-t) (area under the first-moment curve)."""
    moment = time * conc
    return float(np.trapezoid(moment, time))


def _log_segment_ok(c0: float, c1: float) -> bool:
    """Log-trapezoid is only defined for two positive, unequal
    concentrations — every other segment falls back to linear."""
    return c0 > 0 and c1 > 0 and c0 != c1


def _aumc_segments(time: np.ndarray, conc: np.ndarray, use_log) -> float:
    """AUMC summed segment by segment, log-moment formula on segments where
    use_log(c0, c1) holds. Log-moment term is the exact integral of t*C(t)
    for a log-linear C between the two points:
        dt*(t0*C0 - t1*C1)/ln(C0/C1) + dt^2*(C0 - C1)/ln(C0/C1)^2
    """
    total = 0.0
    for i in range(len(time) - 1):
        t0, t1 = float(time[i]), float(time[i + 1])
        c0, c1 = float(conc[i]), float(conc[i + 1])
        dt = t1 - t0
        if use_log(c0, c1):
            ln = np.log(c0 / c1)
            total += dt * (t0 * c0 - t1 * c1) / ln + dt ** 2 * (c0 - c1) / ln ** 2
        else:
            total += (t0 * c0 + t1 * c1) * dt / 2.0
    return float(total)


def aumc_linear_up_log_down(time: np.ndarray, conc: np.ndarray) -> float:
    """AUMC partner of auc_linear_up_log_down — log moment while declining."""
    return _aumc_segments(time, conc, lambda c0, c1: _log_segment_ok(c0, c1) and c1 < c0)


def aumc_linear_log(time: np.ndarray, conc: np.ndarray) -> float:
    """AUMC partner of auc_linear_log — log moment on every valid segment."""
    return _aumc_segments(time, conc, _log_segment_ok)


def lambda_z(time: np.ndarray, conc: np.ndarray, n_points: int | None = None,
             t_range: tuple[float, float] | None = None,
             excluded_times: set[float] | None = None,
             exclude_pre_tmax: bool = False) -> dict:
    """Terminal elimination rate constant via OLS log-linear regression.

    If t_range is given (start, end), fits only points within that time
    window — user-specified Time Range mode, overrides n_points.
    If n_points is None and t_range is None, auto-selects the number of
    terminal points (>= MIN_TERMINAL_POINTS) that maximizes adjusted R^2.
    Auto-selection is a heuristic — callers doing research-grade work
    should let the user override via the UI, never trust this blindly.

    exclude_pre_tmax drops every point at or before Tmax before the Best Fit
    search (Guide p.135) — for extravascular/infusion routes the absorption
    limb is not part of the terminal phase, and letting the search reach
    back into it silently biases lambda_z. Ignored in Time Range and
    explicit-n_points modes: there the user already chose the window.

    excluded_times drops those time points from the regression only —
    callers must still use the original, unfiltered time/conc for
    AUC/AUMC; this function never touches those.
    """
    mask = conc > 0
    t = time[mask]
    c = conc[mask]
    empty = {"lambda_z": None, "half_life": None, "n_points": 0, "r_squared": None,
             "adj_r_squared": None, "slope": None, "intercept": None,
             "terminal_t": None, "terminal_conc": None}

    if exclude_pre_tmax and t_range is None and n_points is None and len(t):
        tmax = t[int(np.argmax(c))]
        keep = t > tmax
        t, c = t[keep], c[keep]

    if excluded_times:
        keep = ~np.isin(t, list(excluded_times))
        t, c = t[keep], c[keep]

    if len(t) < MIN_TERMINAL_POINTS:
        return empty

    log_c = np.log(c)

    def fit_points(tt: np.ndarray, yy: np.ndarray) -> dict:
        slope, intercept = np.polyfit(tt, yy, 1)
        pred = slope * tt + intercept
        ss_res = np.sum((yy - pred) ** 2)
        ss_tot = np.sum((yy - np.mean(yy)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        n = len(tt)
        adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - 2) if n > 2 else r2
        return {"slope": slope, "intercept": intercept, "r2": r2, "adj_r2": adj_r2, "n": n}

    if t_range is not None:
        lo, hi = t_range
        in_range = (t >= lo) & (t <= hi)
        if int(in_range.sum()) < MIN_TERMINAL_POINTS:
            return empty
        best = fit_points(t[in_range], log_c[in_range])
        t_used, c_used = t[in_range], c[in_range]
    elif n_points is not None:
        n = min(n_points, len(t))
        if n < MIN_TERMINAL_POINTS:
            return empty  # same floor the other two modes enforce
        best = fit_points(t[-n:], log_c[-n:])
        t_used, c_used = t[-n:], c[-n:]
    else:
        candidates = [fit_points(t[-n:], log_c[-n:]) for n in range(MIN_TERMINAL_POINTS, len(t) + 1)]
        top = max(candidates, key=lambda r: r["adj_r2"])
        # WinNonlin tie-break (Guide p.136): adding a point has to improve
        # adjusted R^2 by more than 0.0001 to matter — among everything
        # within that tolerance of the best, take the fit using the most
        # points. Plain max() would silently settle for the minimum 3 points
        # on a clean log-linear tail, where every candidate scores 1.0.
        best = max((r for r in candidates if r["adj_r2"] > top["adj_r2"] - 1e-4),
                    key=lambda r: r["n"])
        t_used, c_used = t[-best["n"]:], c[-best["n"]:]

    slope = best["slope"]
    # A flat profile fits a slope of ~1e-16 instead of exactly 0, which would
    # otherwise pass as "declining" and report a half-life of ~1e15 hours.
    # Test the total log-concentration drop across the window rather than the
    # slope itself — that stays scale-free in both time and concentration.
    log_drop = -slope * (t_used[-1] - t_used[0])
    if not np.isfinite(slope) or log_drop < 1e-10:
        return {**empty, "n_points": best["n"], "r_squared": best["r2"],
                "adj_r_squared": best["adj_r2"]}

    lz = -slope
    return {
        "lambda_z": float(lz),
        "half_life": float(np.log(2) / lz),
        "n_points": int(best["n"]),
        "r_squared": float(best["r2"]),
        "adj_r_squared": float(best["adj_r2"]),
        "slope": float(slope),
        "intercept": float(best["intercept"]),
        "terminal_t": t_used.tolist(),
        "terminal_conc": c_used.tolist(),
    }


def compute_flags(n_points: int, lz: dict, pct_extrap: float | None) -> dict:
    """Advisory acceptance-criteria flags (Guide p.130). A profile that
    fails one of these still reports its numbers — flags surface a quality
    concern, they never suppress a result."""
    flag_n_samples = "Insufficient" if n_points < MIN_SAMPLES else "OK"

    if lz["lambda_z"] is None:
        return {
            "flag_n_samples": flag_n_samples,
            "span": None,
            "flag_low_rsq": None,
            "flag_high_extrap": None,
            "flag_low_span": None,
        }

    span = (max(lz["terminal_t"]) - min(lz["terminal_t"])) / lz["half_life"]
    return {
        "flag_n_samples": flag_n_samples,
        "span": float(span),
        "flag_low_rsq": bool(lz["adj_r_squared"] < RSQ_ADJ_THRESHOLD),
        "flag_high_extrap": bool(pct_extrap is not None and pct_extrap > PCT_EXTRAP_THRESHOLD),
        "flag_low_span": bool(span < SPAN_THRESHOLD),
    }


def steady_state_metrics(time: np.ndarray, conc: np.ndarray, tau: float,
                          auc_method: str = "linear", lambda_z_value: float | None = None) -> dict:
    """Steady-state parameters over one dosing interval [0, tau] (Guide
    p.148-150). Assumes the profile's t=0 is the start of that interval.
    Cmax/Cmin/Ctau/Cavg/Fluctuation replace the single-dose Cmax/Tmax at
    steady state; Accumulation is the theoretical ratio from lambda_z,
    independent of any reference single-dose profile."""
    window = (time >= 0) & (time <= tau)
    t_win, c_win = time[window], conc[window]
    idx_max = int(np.argmax(c_win))
    cmax_ss = float(c_win[idx_max])
    tmax_ss = float(t_win[idx_max])
    cmin_ss = float(np.min(c_win))
    ctau = _point_conc(time, conc, tau)

    auc_fn = AUC_METHODS.get(auc_method, auc_linear)
    pts_t, pts_c = _clip_points(time, conc, 0.0, tau)
    auc_tau = float(auc_fn(pts_t, pts_c)) if len(pts_t) else 0.0
    cavg_ss = auc_tau / tau if tau else None

    pct_fluctuation = (cmax_ss - cmin_ss) / cavg_ss * 100.0 if cavg_ss else None
    accumulation_index = (1.0 / (1.0 - np.exp(-lambda_z_value * tau))
                           if lambda_z_value else None)

    return {
        "tau": float(tau),
        "cmax_ss": cmax_ss,
        "tmax_ss": tmax_ss,
        "cmin_ss": cmin_ss,
        "ctau": float(ctau),
        "auc_tau": auc_tau,
        "cavg_ss": cavg_ss,
        "pct_fluctuation": pct_fluctuation,
        "accumulation_index": accumulation_index,
        # Tau past the last sample means AUCtau covers a shorter window than
        # the dosing interval, so Cavg/%Fluctuation are biased low. Reported,
        # never silently swallowed.
        "flag_tau_beyond_tlast": bool(tau > time[-1]),
    }


AUC_METHODS = {
    "linear": auc_linear,
    "linear_up_log_down": auc_linear_up_log_down,
    "linear_log": auc_linear_log,
    "linear_interp": auc_linear_interp,
}

# AUMC must use the same interpolation rule as AUC — a log-down AUC paired
# with a linear AUMC gives an internally inconsistent MRT (and so Vss).
AUMC_METHODS = {
    "linear": aumc_linear,
    "linear_up_log_down": aumc_linear_up_log_down,
    "linear_log": aumc_linear_log,
    "linear_interp": aumc_linear,
}


def nca_iv_bolus(time: np.ndarray, conc: np.ndarray, dose: float, n_terminal: int | None = None,
                  auc_method: str = "linear", lz_t_range: tuple[float, float] | None = None,
                  lz_excluded_times: set[float] | None = None, tau: float | None = None,
                  weight: float | None = None) -> dict:
    return _nca_common(time, conc, dose, n_terminal, auc_method, route="iv_bolus", infusion_duration=0.0,
                        lz_t_range=lz_t_range, lz_excluded_times=lz_excluded_times, tau=tau, weight=weight)


def nca_iv_infusion(time: np.ndarray, conc: np.ndarray, dose: float, infusion_duration: float,
                     n_terminal: int | None = None, auc_method: str = "linear",
                     lz_t_range: tuple[float, float] | None = None,
                     lz_excluded_times: set[float] | None = None, tau: float | None = None,
                     weight: float | None = None) -> dict:
    return _nca_common(time, conc, dose, n_terminal, auc_method, route="iv_infusion", infusion_duration=infusion_duration,
                        lz_t_range=lz_t_range, lz_excluded_times=lz_excluded_times, tau=tau, weight=weight)


def nca_extravascular(time: np.ndarray, conc: np.ndarray, dose: float, n_terminal: int | None = None,
                       auc_method: str = "linear", lz_t_range: tuple[float, float] | None = None,
                       lz_excluded_times: set[float] | None = None, tau: float | None = None,
                       weight: float | None = None) -> dict:
    return _nca_common(time, conc, dose, n_terminal, auc_method, route="extravascular", infusion_duration=0.0,
                        lz_t_range=lz_t_range, lz_excluded_times=lz_excluded_times, tau=tau, weight=weight)


# Every key a successful _nca_common run produces, all None. Kept in sync by
# test_degenerate_result_has_the_same_keys_as_a_normal_one — if a parameter is
# added below without being added here, that test fails.
_NULL_RESULT = {
    "cmax": None, "tmax": None, "clast": None, "tlast": None,
    "lambda_z": None, "half_life": None, "n_points": 0, "r_squared": None,
    "adj_r_squared": None, "slope": None, "intercept": None,
    "terminal_t": None, "terminal_conc": None,
    "auc_t": None, "auc_inf": None, "auc_inf_pred": None,
    "pct_extrap": None, "pct_back_ext": None,
    "aumc_t": None, "aumc_inf": None, "mrt": None,
    "cl": None, "vz": None, "vss": None,
    "dose_per_kg": None, "cl_per_kg": None, "vz_per_kg": None, "vss_per_kg": None,
    "span": None, "flag_low_rsq": None, "flag_high_extrap": None, "flag_low_span": None,
}


def _nca_common(time: np.ndarray, conc: np.ndarray, dose: float, n_terminal, auc_method: str,
                 route: str, infusion_duration: float,
                 lz_t_range: tuple[float, float] | None = None,
                 lz_excluded_times: set[float] | None = None,
                 tau: float | None = None,
                 weight: float | None = None) -> dict:
    time = np.asarray(time, dtype=float)
    conc = np.asarray(conc, dtype=float)

    # BQL / missing / blank cells arrive as NaN. Dropping them here is the
    # only place it can be done once for every route — left in, a single NaN
    # silently turns AUC, AUMC, CL, Vz and every derived parameter into NaN.
    finite = np.isfinite(time) & np.isfinite(conc)
    n_excluded = int((~finite).sum())
    time, conc = time[finite], conc[finite]

    # Guide p.134: profiles are analyzed in ascending time order. Callers
    # usually sort already; a pure function shouldn't rely on that.
    order = np.argsort(time, kind="stable")
    time, conc = time[order], conc[order]

    if len(time) == 0:
        # Same key set as a successful run, every value None. A degenerate
        # profile that returned a *shorter* dict forced every caller to use
        # .get() or risk a KeyError — the results table and the Core Output
        # would silently show fewer rows for it instead of showing blanks.
        return {**_NULL_RESULT, "route": route, "dose": dose,
                "n_excluded": n_excluded, "flag_n_samples": "Insufficient",
                "note": "no usable (finite) concentration-time points"}

    # t=0 insertion (Guide p.134). Which concentration goes in depends on
    # the situation, and getting it wrong silently moves AUC:
    #   IV bolus      -> back-extrapolated C0 (can exceed the first sample)
    #   steady state  -> Cmin, the trough the interval starts from
    #   otherwise     -> 0, since no drug is in plasma before an
    #                    extravascular dose or the start of an infusion
    # Skipping this for non-bolus routes understated early AUC for every
    # profile whose first sample wasn't drawn at t=0.
    pct_back_ext = None
    back_ext_segment = None
    if time[0] != 0:
        if route == "iv_bolus":
            c_at_zero = bolus_c0(time, conc)
            back_ext_segment = (c_at_zero + conc[0]) * time[0] / 2.0
        elif tau is not None:
            # Cmin of the dosing interval, not of the whole record. A profile
            # sampled past Tau (say 24h of data against a 12h interval) would
            # otherwise seed t=0 with a trough from a later interval, which
            # then also becomes the reported Cmin_ss.
            in_interval = conc[time <= tau]
            c_at_zero = float(np.min(in_interval)) if len(in_interval) else float(np.min(conc))
        else:
            c_at_zero = 0.0
        time = np.concatenate(([0.0], time))
        conc = np.concatenate(([c_at_zero], conc))

    result = cmax_tmax(time, conc)

    auc_fn = AUC_METHODS.get(auc_method, auc_linear)
    auc_t = auc_fn(time, conc)
    aumc_t = AUMC_METHODS.get(auc_method, aumc_linear)(time, conc)

    lz = lambda_z(time, conc, n_terminal, t_range=lz_t_range, excluded_times=lz_excluded_times,
                  exclude_pre_tmax=(route != "iv_bolus"))
    result.update(lz)

    clast = float(conc[-1])
    tlast = float(time[-1])

    if lz["lambda_z"] is not None:
        auc_extrap = clast / lz["lambda_z"]
        auc_inf = auc_t + auc_extrap
        aumc_extrap = (clast * tlast) / lz["lambda_z"] + clast / (lz["lambda_z"] ** 2)
        aumc_inf = aumc_t + aumc_extrap

        cl = dose / auc_inf if dose else None
        vz = cl / lz["lambda_z"] if cl is not None else None

        mrt = aumc_inf / auc_inf
        if route == "iv_infusion" and infusion_duration:
            mrt -= infusion_duration / 2.0
        vss = cl * mrt if cl is not None else None

        # AUCINF(pred): extrapolation from the regression line's predicted
        # Clast instead of the observed one.
        clast_pred = float(np.exp(lz["intercept"] + lz["slope"] * tlast))
        auc_inf_pred = auc_t + clast_pred / lz["lambda_z"]
        pct_extrap = auc_extrap / auc_inf * 100.0
    else:
        auc_inf = aumc_inf = cl = vz = vss = mrt = auc_inf_pred = pct_extrap = None

    # %Back_Ext is the back-extrapolated slice as a fraction of AUCINF
    # (WinNonlin definition); AUClast is only the fallback when lambda_z —
    # and so AUCINF — isn't estimable.
    if back_ext_segment is not None:
        denom = auc_inf if auc_inf else auc_t
        pct_back_ext = back_ext_segment / denom * 100.0 if denom else None

    # Weight-normalized (mg/kg) params — only when a body weight is given.
    if weight:
        dose_per_kg = dose / weight
        cl_per_kg = cl / weight if cl is not None else None
        vz_per_kg = vz / weight if vz is not None else None
        vss_per_kg = vss / weight if vss is not None else None
    else:
        dose_per_kg = cl_per_kg = vz_per_kg = vss_per_kg = None

    result.update({
        "route": route,
        "clast": clast,
        "tlast": tlast,
        "auc_t": auc_t,
        "auc_inf": auc_inf,
        "auc_inf_pred": auc_inf_pred,
        "pct_extrap": pct_extrap,
        "pct_back_ext": pct_back_ext,
        "aumc_t": aumc_t,
        "aumc_inf": aumc_inf,
        "cl": cl,
        "vz": vz,
        "vss": vss,
        "mrt": mrt,
        "dose": dose,
        "n_excluded": n_excluded,
        "dose_per_kg": dose_per_kg,
        "cl_per_kg": cl_per_kg,
        "vz_per_kg": vz_per_kg,
        "vss_per_kg": vss_per_kg,
    })
    result.update(compute_flags(len(time), lz, pct_extrap))

    if tau is not None:
        result.update(steady_state_metrics(time, conc, tau, auc_method, lz["lambda_z"]))

    return result


_FLAG_MESSAGES = {
    "flag_n_samples": lambda r: "Insufficient samples for a reliable analysis." if r["flag_n_samples"] == "Insufficient" else None,
    "flag_low_rsq": lambda r: f"Terminal-phase R-squared below {RSQ_ADJ_THRESHOLD} — lambda_z fit may be unreliable." if r.get("flag_low_rsq") else None,
    "flag_high_extrap": lambda r: f"AUC extrapolated fraction above {PCT_EXTRAP_THRESHOLD}% — AUCINF may be unreliable." if r.get("flag_high_extrap") else None,
    "flag_low_span": lambda r: f"Terminal phase spans fewer than {SPAN_THRESHOLD} half-lives — lambda_z estimate may be unreliable." if r.get("flag_low_span") else None,
    "n_excluded": lambda r: f"{r['n_excluded']} sample(s) excluded as missing/BQL (non-numeric time or concentration)." if r.get("n_excluded") else None,
    "flag_tau_beyond_tlast": lambda r: "Tau extends past the last sample — AUCtau covers a shorter window than the dosing interval, so Cavg_ss and %Fluctuation are biased low." if r.get("flag_tau_beyond_tlast") else None,
}


def format_core_output(result: dict, settings: dict, units: dict[str, str] | None = None) -> str:
    """Full settings + results + warnings as one traceable text block
    (Guide's Core Output requirement) — everything needed to reproduce a
    result by hand, in one file."""
    units = units or {}
    lines = ["PKPD Software — NCA Core Output", "=" * 40, "", "Settings:"]
    for key, value in settings.items():
        lines.append(f"  {key}: {value}")

    lines += ["", "Results:"]
    for key, value in result.items():
        if key in ("terminal_t", "terminal_conc"):
            continue
        unit = units.get(key, "")
        lines.append(f"  {key}: {value} {unit}".rstrip())

    warnings = [msg(result) for msg in _FLAG_MESSAGES.values() if msg(result)]
    lines += ["", "Warnings:"]
    lines += [f"  - {w}" for w in warnings] if warnings else ["  none"]

    return "\n".join(lines)
