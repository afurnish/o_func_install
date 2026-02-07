# -*- coding: utf-8 -*-
"""
Tide gauge overlay + lag-corrected residual tester

- Top panel: Tide Gauge + IRENE + UKC4 (clear line hierarchy)
- Bottom panel: residuals (model - obs), optionally after lag correction

Lag correction modes:
  SHIFT_MODE = "none"   -> no correction, show raw model vs obs
  SHIFT_MODE = "auto"   -> estimate best constant lag in MINUTES and apply via interpolation
  SHIFT_MODE = "manual" -> apply fixed lags in minutes (per model)

This is a *tester* script intended to be dropped into your command editor.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#%%
# ============================================================
# USER CONTROLS
# ============================================================
SHIFT_MODE = "manual"         # "none", "auto", or "manual"
MAX_LAG_MINUTES = 180       # only used for SHIFT_MODE="auto" (search +/- this many minutes)
LAG_STEP_MINUTES = 1        # only used for SHIFT_MODE="auto" (1 = minute resolution)
MANUAL_LAG_MINUTES = {      # only used for SHIFT_MODE="manual"
    "IRENE": 25,             # e.g. +20 (positive shifts model FORWARD in time)
    "UKC4": 25,
}

# Window selection:
WINDOW_DAYS = 14            # spring-neap length
MIN_VALID_FRAC = 0.98       # require % finite pairs in window
# ============================================================


def _as_1d_array(x):
    if hasattr(x, "to_numpy"):
        return x.to_numpy()
    return np.asarray(x)


def _infer_dt_minutes(time_index):
    """Infer time step in minutes from numpy datetime64 array."""
    tt = np.asarray(time_index)
    if len(tt) < 2:
        return 60.0
    dt = (tt[1] - tt[0]) / np.timedelta64(1, "m")
    try:
        return float(dt)
    except Exception:
        return 60.0


def rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() == 0:
        return np.nan
    d = a[ok] - b[ok]
    return float(np.sqrt(np.mean(d * d)))


def shift_series_by_minutes(tt, y, shift_minutes):
    """
    Shift a series by arbitrary minutes using linear interpolation.
    Positive shift_minutes means model is moved FORWARD in time (i.e. model happens later).

    Returns y_shift aligned to the *original* tt grid:
      y_shift(t) = y(t - shift)
    """
    tt = np.asarray(tt)
    y = _as_1d_array(y).astype(float)

    t0 = tt[0]
    tmins = ((tt - t0) / np.timedelta64(1, "m")).astype(float)

    # query original y at (t - shift)
    t_query = tmins - float(shift_minutes)

    ok = np.isfinite(y)
    y_shift = np.full_like(y, np.nan, dtype=float)
    if ok.sum() >= 2:
        y_shift[:] = np.interp(t_query, tmins[ok], y[ok], left=np.nan, right=np.nan)
    return y_shift


def best_lag_minutes_by_rmse(tt, obs, mod, max_lag_minutes=180, step_minutes=1):
    """
    Find constant lag (in minutes) that minimises RMSE between shifted model and obs:
      minimise RMSE( shift(mod, lag) - obs )

    Uses interpolation-based shifting (sub-sample capable).
    """
    tt = np.asarray(tt)
    o = _as_1d_array(obs).astype(float)
    m = _as_1d_array(mod).astype(float)

    # valid pairs (for scoring)
    ok0 = np.isfinite(o) & np.isfinite(m)
    if ok0.sum() < 10:
        return 0.0

    lags = np.arange(-max_lag_minutes, max_lag_minutes + step_minutes, step_minutes, dtype=float)

    best_lag = 0.0
    best_score = np.inf

    for lag in lags:
        m_shift = shift_series_by_minutes(tt, m, lag)
        sc = rmse(m_shift, o)
        if np.isfinite(sc) and sc < best_score:
            best_score = sc
            best_lag = float(lag)

    return best_lag


def build_series_for_gauge(list_of_data, gauge_i, variable_name, time_sliced):
    """
    Uses your naming logic:
      Tide Gauge, UKC4_{PRIMEA} -> IRENE, UKC4_{ao} -> UKC4
    """
    if variable_name != "surface_height":
        data_iter = list_of_data[1:]  # remove tide gauge
        mod_key_new = [r"UKC4$_{\mathrm{PRIMEA}}$", r"UKC4$_{\mathrm{ao}}$"]
    else:
        data_iter = list_of_data
        mod_key_new = ["Tide Gauge", r"UKC4$_{\mathrm{PRIMEA}}$", r"UKC4$_{\mathrm{ao}}$"]

    tt = np.asarray(time_sliced)
    series_dict = {}

    for kil, model in enumerate(data_iter):
        mk = list(model[gauge_i].keys())[0]
        s = model[gauge_i][mk]

        if mod_key_new[kil] == r"UKC4$_{\mathrm{PRIMEA}}$":
            new_name = "IRENE"
        elif mod_key_new[kil] == r"UKC4$_{\mathrm{ao}}$":
            new_name = "UKC4"
        else:
            new_name = mod_key_new[kil]

        series_dict[new_name] = s

    return tt, series_dict


def find_best_window(tt, obs, models_dict, days=14, min_valid_frac=0.98):
    """
    Choose 14-day window that minimises mean RMSE across models vs obs (RAW series).
    """
    dt_min = _infer_dt_minutes(tt)
    win_len = int(round((days * 24.0 * 60.0) / dt_min))
    if win_len < 10 or len(tt) <= win_len:
        return 0, len(tt) - 1, np.nan

    o = _as_1d_array(obs).astype(float)
    model_names = list(models_dict.keys())
    model_arrs = {k: _as_1d_array(models_dict[k]).astype(float) for k in model_names}

    n = len(tt)
    scores = np.full(n - win_len, np.nan, dtype=float)

    for s in range(n - win_len):
        e = s + win_len
        oseg = o[s:e]
        ok_obs = np.isfinite(oseg)
        if ok_obs.mean() < min_valid_frac:
            continue

        rmses = []
        for k in model_names:
            mseg = model_arrs[k][s:e]
            ok = ok_obs & np.isfinite(mseg)
            if ok.mean() < min_valid_frac:
                rmses = []
                break
            diff = mseg[ok] - oseg[ok]
            rmses.append(float(np.sqrt(np.mean(diff * diff))))

        if len(rmses) > 0:
            scores[s] = float(np.mean(rmses))

    if np.all(~np.isfinite(scores)):
        return 0, win_len - 1, np.nan

    s0 = int(np.nanargmin(scores))
    s1 = s0 + win_len - 1
    return s0, s1, float(scores[s0])


def preview_best_window_plot(list_of_data, gauge_i, variable_name, time_sliced, gauge_name=None):
    tt, series = build_series_for_gauge(list_of_data, gauge_i, variable_name, time_sliced)

    if "Tide Gauge" not in series:
        raise ValueError("No 'Tide Gauge' series found (variable_name must be 'surface_height').")

    obs = series["Tide Gauge"]
    models = {k: v for k, v in series.items() if k != "Tide Gauge"}

    # select window using RAW series (keeps selection honest / reproducible)
    s0, s1, score = find_best_window(tt, obs, models, days=WINDOW_DAYS, min_valid_frac=MIN_VALID_FRAC)

    tts = np.asarray(tt[s0:s1 + 1])
    obs_s = obs.iloc[s0:s1 + 1] if hasattr(obs, "iloc") else _as_1d_array(obs)[s0:s1 + 1]
    oarr = _as_1d_array(obs_s).astype(float)

    # Clear “examiner-friendly” styling
    STYLE = {
        "Tide Gauge": dict(color="0.10", lw=3.0,  ls="-",  alpha=1.0,  zorder=6),
        "IRENE":      dict(color="red",  lw=2.25, ls="--", alpha=0.95, zorder=4),
        "UKC4":       dict(color="blue", lw=1.75, ls=":",  alpha=0.95, zorder=3),
    }

    fig, (ax, axr) = plt.subplots(
        2, 1, dpi=300, sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
    )
    fig.set_figheight(6.0)
    fig.set_figwidth(10.5)

    # Top: plot observation
    ax.plot(tts, obs_s, label="Tide Gauge", **STYLE["Tide Gauge"], solid_capstyle="round")

    # Bottom: plot residuals (optionally lag-corrected)
    axr.axhline(0.0, linewidth=0.9, linestyle="-", color="0.5", alpha=0.8)

    lag_report = []

    for name, ms in models.items():
        ms_s = ms.iloc[s0:s1 + 1] if hasattr(ms, "iloc") else _as_1d_array(ms)[s0:s1 + 1]
        marr = _as_1d_array(ms_s).astype(float)

        st = STYLE.get(name, dict(color="0.4", lw=2.0, ls="--", alpha=0.9, zorder=2))

        # determine lag in minutes
        if SHIFT_MODE.lower() == "none":
            lag_min = 0.0
        elif SHIFT_MODE.lower() == "manual":
            lag_min = float(MANUAL_LAG_MINUTES.get(name, 0.0))
        else:
            lag_min = best_lag_minutes_by_rmse(
                tts, oarr, marr,
                max_lag_minutes=MAX_LAG_MINUTES,
                step_minutes=LAG_STEP_MINUTES
            )

        # apply lag correction via interpolation
        marr_shift = shift_series_by_minutes(tts, marr, lag_min)

        # TOP: show lag-corrected model (clear overlay)
        ax.plot(tts, marr_shift, label=name, **st, solid_capstyle="round")

        # BOTTOM: show lag-corrected residuals (or raw if SHIFT_MODE="none")
        resid = marr_shift - oarr
        axr.plot(tts, resid, linewidth=1.2, linestyle=st["ls"], color=st["color"], alpha=0.95)

        lag_report.append((name, lag_min, rmse(marr, oarr), rmse(marr_shift, oarr)))

    # Formatting / readability
    ax.grid(True, which="major", linewidth=0.3, alpha=0.4)
    axr.grid(True, which="major", linewidth=0.3, alpha=0.4)

    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.22),
        ncol=3, frameon=True, framealpha=0.9
    )

    ax.set_ylabel("Surface Height [m]")
    if SHIFT_MODE.lower() == "none":
        axr.set_ylabel("Model - Obs [m]")
    else:
        axr.set_ylabel("Lag-corrected\nModel - Obs [m]")

    axr.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    axr.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()

    title_bits = []
    if gauge_name:
        title_bits.append(str(gauge_name))
    if np.isfinite(score):
        title_bits.append(f"best {WINDOW_DAYS}-day window (raw selection) mean RMSE={score:.3f} m")
    title_bits.append(f"lag mode={SHIFT_MODE}")
    ax.set_title(" | ".join(title_bits))

    plt.tight_layout()
    plt.show()

    # Console output
    print(f"Best window indices: {s0} -> {s1} (len={s1 - s0 + 1})  score={score}")
    print(f"Best window dates  : {pd.Timestamp(tt[s0]).date()} -> {pd.Timestamp(tt[s1]).date()}")
    for (name, lag_min, rm_raw, rm_corr) in lag_report:
        if SHIFT_MODE.lower() == "none":
            print(f"{gauge_name} | {name}: RMSE raw={rm_raw:.3f} m")
        else:
            print(f"{gauge_name} | {name}: lag={lag_min:+.0f} min | RMSE raw={rm_raw:.3f} m | RMSE lag-corr={rm_corr:.3f} m")

    return s0, s1, score


# ---- Call (as in your original tester usage) ----
# Assumes you are running this inside your existing context where variable_name, self, i, list_of_data exist
if variable_name == "surface_height":
    tide_gauge_name = [j for j in self.tide_loc_dict.keys()][i]
    preview_best_window_plot(list_of_data, i, variable_name, self.time_sliced, gauge_name=tide_gauge_name)
