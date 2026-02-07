#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 14:18:54 2025

@author: af
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.dates as mdates
from o_func import opsys
import pandas as pd
from scipy.signal import butter, filtfilt

'''
At these cells
   mesh2d_face_x = -3.22601205110508
   mesh2d_face_y = 53.9555096590014
'''

start_path = Path(opsys('PNC'))

infile = start_path / 'modelling_DATA/kent_estuary_project/diffusion_viscosity/centre_point/centre_point_timeseries.nc'
outdir = infile.parent / "fig_timeseries"
outdir.mkdir(parents=True, exist_ok=True)

time_name = "time"
wind_name = "wind_speed"                 # m/s
tke_name  = "tke_surface_edge_mean"  
# --- Load reduced dataset ---

# ds = xr.open_dataset(single_point_path)

def load_series(ds: xr.Dataset):
    """Return dict of 1D time series with consistent names, handling both formats."""
    out = {}

    # Preferred (reduced file)
    if set([
        "wind_speed",
        "salinity_surface",
        "salinity_bottom",
        "salinity_delta_surface_minus_bottom",
        "tke_surface_edge_mean",
        "vic_surface_edge_mean",
    ]).issubset(ds.data_vars):
        out["time"] = ds["time"]
        out["wind"] = ds["wind_speed"]
        out["tke"]  = ds["tke_surface_edge_mean"]
        out["vic"]  = ds["vic_surface_edge_mean"]
        out["S_surf"]  = ds["salinity_surface"]
        out["S_bot"]   = ds["salinity_bottom"]
        out["dS"]      = ds["salinity_delta_surface_minus_bottom"]
        return out

    # Fallback (full map file): compute from mesh2d_* at a face/edge
    # Expect wind, salinity, tke/vic with dimensions. User must select face & interface.
    # Try to pick a single face from attributes if present
    face_idx = None
    if "face_index" in ds:
        try:
            face_idx = int(ds["face_index"].values)
        except Exception:
            face_idx = None

    # Try to guess a face: if 2D var has dims ('time','mesh2d_nFaces'), choose first
    def pick_face(da):
        nonlocal face_idx
        if face_idx is not None and "mesh2d_nFaces" in da.dims:
            return da.isel(mesh2d_nFaces=face_idx)
        if "mesh2d_nFaces" in da.dims:
            return da.isel(mesh2d_nFaces=0)
        return da

    time = ds["time"]

    if {"mesh2d_windx","mesh2d_windy"}.issubset(ds.data_vars):
        wx = pick_face(ds["mesh2d_windx"])
        wy = pick_face(ds["mesh2d_windy"])
        wind = np.sqrt(wx**2 + wy**2)
    elif "wind_speed" in ds:
        wind = ds["wind_speed"]
    else:
        raise ValueError("Could not find wind variables.")

    # Salinity: surface/bottom layer on faces
    if "mesh2d_sa1" in ds:
        sa = pick_face(ds["mesh2d_sa1"])  # (time, mesh2d_nLayers) after pick_face
        if "mesh2d_nLayers" in sa.dims:
            S_surf = sa.isel(mesh2d_nLayers=0)
            S_bot  = sa.isel(mesh2d_nLayers=-1)
            dS     = S_surf - S_bot
        else:
            # single-layer file
            S_surf = sa
            S_bot  = sa
            dS     = xr.zeros_like(sa)
    else:
        # last resort: look for already-derived names
        S_surf = ds.get("salinity_surface")
        S_bot  = ds.get("salinity_bottom", S_surf)
        if S_surf is None:
            raise ValueError("No salinity found.")
        dS = S_surf - S_bot

    # TKE/VIC: interfaces on edges — just take surface interface (0) and mean over edges if present
    def edge_surface_mean(name):
        if name not in ds:
            return None
        da = ds[name]
        if "mesh2d_nInterfaces" in da.dims:
            da = da.isel(mesh2d_nInterfaces=0)
        if "mesh2d_nEdges" in da.dims:
            da = da.mean("mesh2d_nEdges", skipna=True)
        return da

    if "tke_surface_edge_mean" in ds:
        tke = ds["tke_surface_edge_mean"]
    else:
        tke = edge_surface_mean("mesh2d_turkin1")

    if "vic_surface_edge_mean" in ds:
        vic = ds["vic_surface_edge_mean"]
    else:
        vic = edge_surface_mean("mesh2d_vicwwu")

    return {
        "time": time,
        "wind": wind,
        "tke": tke,
        "vic": vic,
        "S_surf": S_surf,
        "S_bot": S_bot,
        "dS": dS,
    }

# ----------- run -----------
ds = xr.open_dataset(infile)
ds = ds.sel(time=slice("2013-11-27", None))  # keep from 21 Nov 2013 onward
series = load_series(ds)
time = series["time"]

# Overlay
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111)
ax.plot(time, series["wind"], label='Wind speed (m/s)')
if series["tke"] is not None:
    ax.plot(time, series["tke"], label='TKE (m²/s²)')
if series["vic"] is not None:
    ax.plot(time, series["vic"], label='Vert. Eddy Viscosity (m²/s)')
ax.plot(time, series["dS"], label='ΔS Surface-Bottom (psu)')
ax.legend()
ax.set_title("Overlay of wind, turbulence, and salinity difference")
ax.grid(True)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(outdir / "overlay_timeseries.png", dpi=200)
plt.show()

# Stacked
fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
axs[0].plot(time, series["wind"]); axs[0].set_ylabel("Wind (m/s)")
axs[1].plot(time, series["tke"], color='orange'); axs[1].set_ylabel("TKE (m²/s²)")
axs[2].plot(time, series["vic"], color='green'); axs[2].set_ylabel("VIC (m²/s)")
axs[3].plot(time, series["dS"], color='purple'); axs[3].set_ylabel("ΔS (psu)")
for ax in axs:
    ax.grid(True)
axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
fig.autofmt_xdate()
plt.suptitle("Wind-driven mixing indicators at centre point", y=0.99)
plt.tight_layout()
fig.savefig(outdir / "stacked_timeseries.png", dpi=200)
plt.show()

#%% 
w = ds['wind_speed'].to_series()
t = ds['tke_surface_edge_mean'].to_series()

print("Zero-lag corr:", w.corr(t))

# Try lags (wind leading TKE)
for hrs in (0, 1, 3, 6, 12):
    print(f"Lag {hrs} h:", w.corr(t.shift(-hrs*3)))  # 20-min data ⇒ 3 steps/hour

#%% 

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# If your data are strictly hourly already, leave these alone.
# Otherwise we resample to hourly means.
RESAMPLE_TO_HOURLY = False

# Filter settings (hourly sampling):
#   - High-pass: remove periods > 3 days
#   - Band-pass: keep 6–72 h (captures stormy / synoptic + removes diurnal-semi-diurnal)
HP_CUTOFF_H  = 72.0
BP_LOW_H     = 6.0
BP_HIGH_H    = 72.0

# Rolling correlation window (hours)
ROLL_WIN_H = 48

# Event composite threshold (95th percentile wind)
EVENT_PCTL = 95
EVENT_LEAD_H = 36  # examine response out to +36 h
# --------------------------------

def butter_filter(series, kind="highpass", dt_hours=1.0,
                  hp_cut_h=72.0, bp_low_h=6.0, bp_high_h=72.0, order=3):
    """Zero-phase Butterworth filter on a pandas Series (regular hourly index)."""
    x = series.values.astype(float)
    fs = 1.0 / dt_hours  # samples per hour
    nyq = 0.5 * fs
    if kind == "highpass":
        wc = (1.0/hp_cut_h) / nyq
        b, a = butter(order, wc, btype='highpass')
    elif kind == "bandpass":
        wlow = (1.0/bp_high_h) / nyq
        whigh = (1.0/bp_low_h) / nyq
        b, a = butter(order, [wlow, whigh], btype='bandpass')
    else:
        raise ValueError("kind must be 'highpass' or 'bandpass'")
    y = filtfilt(b, a, x, method="pad")
    return pd.Series(y, index=series.index)

def standardize(s):
    return (s - s.mean()) / s.std(ddof=0)

def to_hourly(ds, name):
    s = ds[name].to_series()
    s = s.sort_index()
    if RESAMPLE_TO_HOURLY:
        s = s.resample("1H").mean()
    # drop obvious NaNs
    return s.dropna()

def cross_corr_at_lags(x, y, max_lag_h=72):
    """Return lags (h) and correlation where positive lag means wind leads TKE."""
    # Align
    df = pd.concat([x, y], axis=1, join="inner").dropna()
    x = df.iloc[:,0]
    y = df.iloc[:,1]
    lags = np.arange(-max_lag_h, max_lag_h+1, 1, dtype=int)
    corrs = []
    for lg in lags:
        if lg > 0:
            corr = x.corr(y.shift(-lg))
        elif lg < 0:
            corr = x.shift(+(-lg)).corr(y)
        else:
            corr = x.corr(y)
        corrs.append(corr)
    return lags, np.array(corrs, float)

# ---------- LOAD ----------
w_raw = to_hourly(ds, wind_name)
t_raw = to_hourly(ds, tke_name)

# Align time & keep common
df_raw = pd.concat([w_raw.rename("wind"), t_raw.rename("tke")], axis=1).dropna()
wind = df_raw["wind"]
tke  = df_raw["tke"]

# ---------- FILTERS ----------
# Use BOTH a high-pass and a 6–72 h band-pass; pick the one that looks cleaner
wind_hp = butter_filter(wind, kind="highpass", hp_cut_h=HP_CUTOFF_H)
tke_hp  = butter_filter(tke,  kind="highpass", hp_cut_h=HP_CUTOFF_H)

wind_bp = butter_filter(wind, kind="bandpass", bp_low_h=BP_LOW_H, bp_high_h=BP_HIGH_H)
tke_bp  = butter_filter(tke,  kind="bandpass", bp_low_h=BP_LOW_H, bp_high_h=BP_HIGH_H)

# Standardize for fair comparison
wind_hp_z, tke_hp_z = standardize(wind_hp), standardize(tke_hp)
wind_bp_z, tke_bp_z = standardize(wind_bp), standardize(tke_bp)

# ---------- CROSS-CORRELATION ----------
lags_h, c_hp = cross_corr_at_lags(wind_hp_z, tke_hp_z, max_lag_h=72)
lags_h2, c_bp = cross_corr_at_lags(wind_bp_z, tke_bp_z, max_lag_h=72)

best_idx_hp = int(np.nanargmax(np.abs(c_hp)))
best_lag_hp = int(lags_h[best_idx_hp])
best_corr_hp = c_hp[best_idx_hp]

best_idx_bp = int(np.nanargmax(np.abs(c_bp)))
best_lag_bp = int(lags_h2[best_idx_bp])
best_corr_bp = c_bp[best_idx_bp]

print(f"[High-pass]  Max |corr|={best_corr_hp:.3f} at lag {best_lag_hp:+d} h "
      "(positive = wind leads)")
print(f"[Band-pass]  Max |corr|={best_corr_bp:.3f} at lag {best_lag_bp:+d} h "
      "(positive = wind leads)")

# ---------- ROLLING CORRELATION (at best band-pass lag) ----------
lag = best_lag_bp
x = wind_bp_z
y = tke_bp_z.shift(-lag) if lag > 0 else tke_bp_z.shift(+abs(lag)) if lag < 0 else tke_bp_z
roll = x.rolling(f"{ROLL_WIN_H}H").corr(y)

# ---------- EVENT COMPOSITES ----------
thr = np.nanpercentile(wind, EVENT_PCTL)
event_starts = wind[(wind >= thr)].index

# Avoid events too close together; keep first after a 24h gap
kept = []
last = None
for t0 in event_starts:
    if last is None or (t0 - last).total_seconds()/3600 >= 24:
        kept.append(t0)
        last = t0
event_starts = kept

# Build composite TKE response following events
hrs = np.arange(0, EVENT_LEAD_H+1, 1)
stacks = []
for t0 in event_starts:
    seg = tke_bp_z.reindex(pd.date_range(t0, t0 + pd.Timedelta(hours=EVENT_LEAD_H), freq="1H"))
    if seg.notna().sum() == len(seg):
        stacks.append(seg.values)
stacks = np.array(stacks) if stacks else np.empty((0, len(hrs)))
comp_mean = stacks.mean(axis=0) if stacks.size else np.full(len(hrs), np.nan)
comp_n = stacks.shape[0]

# ---------- PLOTS ----------
plt.figure(figsize=(10,5))
plt.plot(lags_h, c_hp, label="High-pass")
plt.plot(lags_h2, c_bp, label="Band-pass", linestyle="--")
plt.axvline(0, alpha=0.3)
plt.axhline(0, alpha=0.3)
plt.scatter([best_lag_bp], [best_corr_bp], zorder=5)
plt.title("Wind → TKE cross-correlation (hourly; positive lag = wind leads)")
plt.xlabel("Lag (hours)")
plt.ylabel("Correlation")
plt.legend()
plt.tight_layout()
plt.savefig(outdir / "01_ccf_lag.png", dpi=200)

# Scatter for band-pass anomalies at best lag
x_sc = wind_bp_z
y_sc = tke_bp_z.shift(-best_lag_bp) if best_lag_bp > 0 else \
       tke_bp_z.shift(+abs(best_lag_bp)) if best_lag_bp < 0 else tke_bp_z
df_sc = pd.concat([x_sc, y_sc], axis=1, keys=["wind_bp_z","tke_bp_z_lag"]).dropna()
r_sc = df_sc.corr().iloc[0,1]

plt.figure(figsize=(5.6,5.6))
plt.scatter(df_sc["wind_bp_z"], df_sc["tke_bp_z_lag"], s=10, alpha=0.5)
m, b = np.polyfit(df_sc["wind_bp_z"], df_sc["tke_bp_z_lag"], 1)
xx = np.linspace(df_sc["wind_bp_z"].min(), df_sc["wind_bp_z"].max(), 100)
plt.plot(xx, m*xx + b)
plt.title(f"Band-pass anomalies at lag {best_lag_bp:+d} h (r={r_sc:.2f})")
plt.xlabel("Wind (z-score, 6–72 h)")
plt.ylabel("TKE (z-score, lagged)")
plt.tight_layout()
plt.savefig(outdir / "02_scatter_bandpass.png", dpi=200)

# Rolling correlation over time
plt.figure(figsize=(10,4))
roll.plot()
plt.title(f"Rolling correlation (window={ROLL_WIN_H} h) at lag {best_lag_bp:+d} h")
plt.ylabel("r")
plt.xlabel("Time")
plt.tight_layout()
plt.savefig(outdir / "03_rolling_corr.png", dpi=200)

# Event composite
plt.figure(figsize=(6.5,4.2))
plt.plot(hrs, comp_mean, marker="o")
plt.axhline(0, alpha=0.3)
plt.title(f"TKE band-pass anomaly after strong wind (≥P{EVENT_PCTL}, n={comp_n})")
plt.xlabel("Hours since event start")
plt.ylabel("Mean TKE anomaly (z)")
plt.tight_layout()
plt.savefig(outdir / "04_event_composite.png", dpi=200)

print(f"Saved figures in: {outdir}")

#%%

# Select storm month only
storm_start = pd.Timestamp("2013-12-02")
storm_end   = pd.Timestamp("2013-12-06")

mask = (wind.index >= storm_start) & (wind.index < storm_end)

wind_storm = wind_bp_z[mask]
tke_storm  = tke_bp_z[mask]

# Correlation at lag 0
storm_corr = wind_storm.corr(tke_storm)
print(f"Dec 2013 correlation (lag 0 h, band-pass): {storm_corr:.3f}")

# Full lag scan just for December storms
lags_h, c_storm = cross_corr_at_lags(wind_storm, tke_storm, max_lag_h=72)

plt.figure(figsize=(10,5))
plt.plot(lags_h, c_storm, label="Band-pass Dec storms")
plt.axvline(0, color='grey', alpha=0.5)
plt.axhline(0, color='grey', alpha=0.5)
plt.xlabel("Lag (hours, positive = wind leads)")
plt.ylabel("Correlation")
plt.title("Wind → TKE correlation (Dec 2013 storm period)")
plt.legend()
plt.tight_layout()
plt.show()

#%% 
from scipy.stats import pearsonr, linregress
from pandas import Timestamp, Timedelta
# Band-pass settings for hourly data
BP_LOW_H  = 6.0
BP_HIGH_H = 12
HP_CUT_H  = 12      # only used if you switch to 'highpass'

MAX_LAG_H = 12         # scan ±72 h for max |corr|
RESAMPLE_TO_HOURLY = True

storm_buffer_before = Timedelta(hours=24)
storm_buffer_after  = Timedelta(hours=48)

# Storm windows (inclusive of start, exclusive of end)
# Letter, start (YYYY-MM-DD HH), end (YYYY-MM-DD HH)
storms = [
    ("A", "2013-12-05 00:00", "2013-12-07 00:00"),  # 5–6 Dec
    ("B", "2013-12-18 00:00", "2013-12-20 00:00"),  # 18–19 Dec
    ("C", "2013-12-23 00:00", "2013-12-25 00:00"),  # 23–24 Dec
    ("D", "2013-12-26 00:00", "2013-12-28 00:00"),  # 26–27 Dec
    ("E", "2013-12-30 00:00", "2014-01-01 00:00"),  # 30–31 Dec
    ("F", "2014-01-03 00:00", "2014-01-04 23:59"),  # 3 Jan (your list said 20130103; assumed 2014)
    ("G", "2014-01-05 00:00", "2014-01-06 23:59"),  # 5 Jan (assumed 2014)
]
# ------------------------------------------------

def butter_bandpass(series, dt_hours=1.0, low_h=6.0, high_h=72.0, order=3):
    x = series.values.astype(float)
    fs = 1.0 / dt_hours
    nyq = 0.5 * fs
    wlow = (1.0 / high_h) / nyq
    whigh = (1.0 / low_h) / nyq
    b, a = butter(order, [wlow, whigh], btype='bandpass')
    y = filtfilt(b, a, x, method="pad")
    return pd.Series(y, index=series.index)

def standardize(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)

def to_hourly(ds: xr.Dataset, var: str) -> pd.Series:
    s = ds[var].to_series().sort_index()
    if RESAMPLE_TO_HOURLY:
        s = s.resample("1h").mean()
    return s.dropna()

def cross_corr_lags(x: pd.Series, y: pd.Series, max_lag_h: int):
    """
    Positive lag => wind leads TKE.
    Returns lags array (hours) and correlation array.
    """
    df = pd.concat([x, y], axis=1, join="inner").dropna()
    x = df.iloc[:, 0]; y = df.iloc[:, 1]
    lags = np.arange(-max_lag_h, max_lag_h + 1, dtype=int)
    corrs = np.full_like(lags, np.nan, dtype=float)
    for i, lg in enumerate(lags):
        if lg > 0:
            corrs[i] = x.corr(y.shift(-lg))
        elif lg < 0:
            corrs[i] = x.shift(+(-lg)).corr(y)
        else:
            corrs[i] = x.corr(y)
    return lags, corrs

# ---------- LOAD & PREP ----------
wind_raw = to_hourly(ds, wind_name)
tke_raw  = to_hourly(ds, tke_name)

# Align
df_raw = pd.concat([wind_raw.rename("wind"), tke_raw.rename("tke")], axis=1).dropna()
wind = df_raw["wind"]
tke  = df_raw["tke"]

# Band-pass anomalies (6–72 h) and z-scores
wind_bp = butter_bandpass(wind, low_h=BP_LOW_H, high_h=BP_HIGH_H)
tke_bp  = butter_bandpass(tke,  low_h=BP_LOW_H, high_h=BP_HIGH_H)
wind_bp_z = standardize(wind_bp)
tke_bp_z  = standardize(tke_bp)

# ---------- PER-STORM ANALYSIS ----------
rows = []
for letter, start, end in storms:
    # When looping storms:
    start = Timestamp(start) - storm_buffer_before
    end   = Timestamp(end) + storm_buffer_after
    sel = (wind_bp_z.index >= start) & (wind_bp_z.index < end)
    w = wind_bp_z[sel]
    t = tke_bp_z[sel]
    
    # Require at least, say, 24 hourly samples to be meaningful
    if len(w.dropna()) < 24 or len(t.dropna()) < 24:
        rows.append({
            "Storm": letter,
            "Start": start, "End": end,
            "N": int(min(len(w.dropna()), len(t.dropna()))),
            "MeanWind(m/s)": float(wind[sel].mean()) if sel.any() else np.nan,
            "MeanTKE(m2s2)": float(tke[sel].mean()) if sel.any() else np.nan,
            "r_zero": np.nan, "r_max": np.nan, "lag_at_rmax_h": np.nan,
            "p_at_rmax": np.nan, "slope_at_rmax": np.nan
        })
        continue

    # Zero-lag correlation
    r0, p0 = pearsonr(w.dropna().align(t.dropna(), join="inner")[0],
                      w.dropna().align(t.dropna(), join="inner")[1])

    # Lag scan
    lags, corrs = cross_corr_lags(w, t, MAX_LAG_H)
    if np.all(np.isnan(corrs)):
        rmax, lag_best = np.nan, np.nan
        p_best, slope = np.nan, np.nan
    else:
        # Max absolute correlation
        idx = int(np.nanargmax(np.abs(corrs)))
        rmax = float(corrs[idx])
        lag_best = int(lags[idx])

        # Build lagged pair for stats and slope (TKE vs lagged WIND)
        if lag_best > 0:
            x = w
            y = t.shift(-lag_best)
        elif lag_best < 0:
            x = w.shift(+abs(lag_best))
            y = t
        else:
            x, y = w, t
        pair = pd.concat([x, y], axis=1).dropna()
        if len(pair) >= 10:
            r_best, p_best = pearsonr(pair.iloc[:,0], pair.iloc[:,1])
            reg = linregress(pair.iloc[:,0], pair.iloc[:,1])
            slope = float(reg.slope)  # z-units: 1 std wind → slope std TKE
        else:
            r_best, p_best, slope = np.nan, np.nan, np.nan

    rows.append({
        "Storm": letter,
        "Start": start, "End": end,
        "N": int(len(w.dropna().align(t.dropna(), join="inner")[0])),
        "MeanWind(m/s)": float(wind[sel].mean()),
        "MeanTKE(m2s2)": float(tke[sel].mean()),
        "r_zero": float(r0),
        "r_max": rmax,
        "lag_at_rmax_h": lag_best,
        "p_at_rmax": p_best,
        "slope_at_rmax": slope
    })

summary = pd.DataFrame(rows).sort_values("Start")
print(summary.to_string(index=False))

# Save CSV
csv_path = outdir / "wind_TKE_storm_summary.csv"
summary.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")


#%% Finalised graphs 
# =======================
# Helpers & Parameters
# =======================
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr, linregress

RESAMPLE_TO_HOURLY = True

# Filtering (hourly sampling)
BP_LOW_H  = 6.0     # keep variability faster than ~3/4 day
BP_HIGH_H = 72.0    # up to 3 days (captures synoptic/storm scales)

# Lags (report with positive = wind leads)
MAX_LAG_H = 24

# Storm buffers (extend each event window)
BUFFER_BEFORE_H = 24   # pre-conditioning
BUFFER_AFTER_H  = 48   # decay/re-stratification

def to_hourly(ds: xr.Dataset, var: str) -> pd.Series:
    s = ds[var].to_series().sort_index()
    if RESAMPLE_TO_HOURLY:
        s = s.resample("1h").mean()
    return s.dropna()

def butter_bandpass(series: pd.Series, low_h=6.0, high_h=72.0, order=3) -> pd.Series:
    x = series.values.astype(float)
    fs = 1.0  # samples per hour (after resample)
    nyq = 0.5 * fs
    wlow = (1.0 / high_h) / nyq
    whigh = (1.0 / low_h)  / nyq
    b, a = butter(order, [wlow, whigh], btype='bandpass')
    y = filtfilt(b, a, x, method="pad")
    return pd.Series(y, index=series.index)

def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)

def cross_corr_lags(w: pd.Series, t: pd.Series, max_lag_h=24):
    """Return arrays (lags, corr) with positive lag meaning wind leads TKE."""
    df = pd.concat([w, t], axis=1, join="inner").dropna()
    if df.empty:
        lags = np.arange(-max_lag_h, max_lag_h+1, dtype=int)
        return lags, np.full_like(lags, np.nan, dtype=float)
    w, t = df.iloc[:,0], df.iloc[:,1]
    lags = np.arange(-max_lag_h, max_lag_h+1, dtype=int)
    corrs = []
    for lg in lags:
        if lg > 0:   # wind leads
            corrs.append(w.corr(t.shift(-lg)))
        elif lg < 0: # TKE leads
            corrs.append(w.shift(+(-lg)).corr(t))
        else:
            corrs.append(w.corr(t))
    return lags, np.asarray(corrs, float)

def best_lag_corr(w_z: pd.Series, t_z: pd.Series, max_lag_h=24):
    """Find best |corr| and return (r_best, lag_best, p_value, slope)."""
    lags, corrs = cross_corr_lags(w_z, t_z, max_lag_h=max_lag_h)
    if np.all(np.isnan(corrs)):
        return np.nan, np.nan, np.nan, np.nan
    idx = int(np.nanargmax(np.abs(corrs)))
    lag_best = int(lags[idx])
    # Build matched pair at this lag (positive = wind leads)
    if lag_best > 0:
        x = w_z
        y = t_z.shift(-lag_best)
    elif lag_best < 0:
        x = w_z.shift(+abs(lag_best))
        y = t_z
    else:
        x, y = w_z, t_z
    pair = pd.concat([x, y], axis=1).dropna()
    if len(pair) < 10:
        return float(corrs[idx]), lag_best, np.nan, np.nan
    r, p = pearsonr(pair.iloc[:,0], pair.iloc[:,1])
    reg = linregress(pair.iloc[:,0], pair.iloc[:,1])
    return r, lag_best, p, reg.slope  # slope in z–z units

# =======================
# Load & Prepare
# =======================
wind_raw = to_hourly(ds, wind_name)
tke_raw  = to_hourly(ds, tke_name)
dS_raw   = to_hourly(ds, dS_name) if ('dS_name' in globals() and dS_name) and (dS_name in ds) else None

# Align base frame
df = pd.concat([wind_raw.rename("wind"),
                tke_raw.rename("tke"),
                dS_raw.rename("dS") if dS_raw is not None else None], axis=1).dropna(subset=["wind","tke"])

wind = df["wind"]
tke  = df["tke"]
dS   = df["dS"] if "dS" in df else None

# Band-pass + z-score
wind_bp = butter_bandpass(wind, low_h=BP_LOW_H, high_h=BP_HIGH_H)
tke_bp  = butter_bandpass(tke,  low_h=BP_LOW_H, high_h=BP_HIGH_H)
wind_z, tke_z = zscore(wind_bp), zscore(tke_bp)
dS_z = zscore(butter_bandpass(dS, low_h=BP_LOW_H, high_h=BP_HIGH_H)) if dS is not None else None


# =======================
# Build DISCRETE storm windows (NO buffers) + non-overlap enforcement + mask
# =======================
# storms is your original list of (letter, start, end) with core periods only.
# Example:
# storms = [
#   ("A","2013-12-05 00:00","2013-12-07 00:00"),
#   ("B","2013-12-18 00:00","2013-12-20 00:00"),
#   ...
# ]

# Parse & sort
storm_windows = [(L, pd.Timestamp(s0), pd.Timestamp(s1)) for L, s0, s1 in storms]
storm_windows.sort(key=lambda x: x[1])

# Enforce strictly non-overlapping discrete windows
nonoverlap = []
prev_end = None
for L, S, E in storm_windows:
    if prev_end is not None and S < prev_end:
        S = prev_end  # push start to the previous end if needed
    if S >= E:
        continue  # skip degenerate interval
    nonoverlap.append((L, S, E))
    prev_end = E

storm_windows = nonoverlap  # now discrete & non-overlapping

# Boolean mask over the full time index: union of all discrete windows
storm_mask = pd.Series(False, index=wind.index)
for _, S, E in storm_windows:
    storm_mask.loc[(wind.index >= S) & (wind.index < E)] = True

# =======================
# Stats: full vs storm-only
# =======================
def summarize_block(name, w_z, t_z):
    # zero-lag
    pair0 = pd.concat([w_z, t_z], axis=1).dropna()
    r0, p0 = (pearsonr(pair0.iloc[:,0], pair0.iloc[:,1]) if len(pair0) >= 10 else (np.nan, np.nan))
    # best-lag
    r_best, lag_best, p_best, slope = best_lag_corr(w_z, t_z, max_lag_h=MAX_LAG_H)
    return dict(Block=name, N=len(pair0), r0=r0, p0=p0, r_best=r_best,
                lag_best_h=lag_best, p_best=p_best, slope_z=slope)

full_stats  = summarize_block("Full period",  wind_z, tke_z)
storm_stats = summarize_block("Storm periods (buffered)", wind_z[storm_mask], tke_z[storm_mask])

# Per-storm stats (DISCRETE windows)
per_storm_rows = []
for letter, start, end in storm_windows:
    sel = (wind.index >= start) & (wind.index < end)
    per_storm_rows.append(
        summarize_block(f"Storm {letter} [{start:%Y-%m-%d} to {end:%Y-%m-%d}]",
                        wind_z[sel], tke_z[sel])
    )




# Save summary table
summary_df = pd.DataFrame([full_stats, storm_stats] + per_storm_rows)
summary_csv = outdir / "wind_tke_summary_stats.csv"
summary_df.to_csv(summary_csv, index=False)

# =======================
# Figure with subplots
# =======================
# Best-lag alignments for plotting the scatters
def lag_apply(w_z, t_z, lag_h):
    if np.isnan(lag_h):
        return pd.concat([w_z, t_z], axis=1).dropna()
    lag_h = int(lag_h)
    if lag_h > 0:
        pair = pd.concat([w_z, t_z.shift(-lag_h)], axis=1).dropna()
    elif lag_h < 0:
        pair = pd.concat([w_z.shift(+abs(lag_h)), t_z], axis=1).dropna()
    else:
        pair = pd.concat([w_z, t_z], axis=1).dropna()
    pair.columns = ["wind_z", "tke_z"]
    return pair

pair_full  = lag_apply(wind_z, tke_z, full_stats["lag_best_h"])
pair_storm = lag_apply(wind_z[storm_mask], tke_z[storm_mask], storm_stats["lag_best_h"])

# Time series to plot (wind, TKE, optional dS) with storms shaded
t0, t1 = wind.index.min(), wind.index.max()

fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
ax1 = fig.add_subplot(gs[0,0])  # scatter full
ax2 = fig.add_subplot(gs[0,1])  # scatter storm-only
ax3 = fig.add_subplot(gs[1,:])  # time series

# (1) Scatter: Full period (at best lag)
if len(pair_full) >= 10:
    ax1.scatter(pair_full["wind_z"], pair_full["tke_z"], s=10, alpha=0.5)
    m, b = np.polyfit(pair_full["wind_z"], pair_full["tke_z"], 1)
    xx = np.linspace(pair_full["wind_z"].min(), pair_full["wind_z"].max(), 100)
    ax1.plot(xx, m*xx + b)
ax1.set_title(f"Full period (best lag = {full_stats['lag_best_h']} h, r = {full_stats['r_best']:.2f})")
ax1.set_xlabel("Wind (z, 6–72 h)")
ax1.set_ylabel("TKE (z, 6–72 h)")
ax1.grid(True, alpha=0.3)

# (2) Scatter: Storm-only (at best lag)
if len(pair_storm) >= 10:
    ax2.scatter(pair_storm["wind_z"], pair_storm["tke_z"], s=10, alpha=0.5)
    m2, b2 = np.polyfit(pair_storm["wind_z"], pair_storm["tke_z"], 1)
    xx2 = np.linspace(pair_storm["wind_z"].min(), pair_storm["wind_z"].max(), 100)
    ax2.plot(xx2, m2*xx2 + b2)
ax2.set_title(f"Storm periods (best lag = {storm_stats['lag_best_h']} h, r = {storm_stats['r_best']:.2f})")
ax2.set_xlabel("Wind (z, 6–72 h)")
ax2.set_ylabel("TKE (z, 6–72 h)")
ax2.grid(True, alpha=0.3)

# (3) Time series with storms shaded
ax3.plot(wind.index, wind_z, label="Wind (z, 6–72 h)")
ax3.plot(tke.index,  tke_z,  label="TKE (z, 6–72 h)")
if dS_z is not None:
    ax3.plot(dS_z.index, dS_z, label="ΔS (z, 6–72 h)")
# Shade storms
# Shade storms: light fill + subtle border (one block per discrete storm)
for i, (lab, S, E) in enumerate(storm_windows):
    ax3.axvspan(S, E, facecolor='0.85', alpha=0.7,
                edgecolor='0.5', linewidth=0.6,
                label="Defined storm periods" if i == 0 else None)
ax3.set_xlim(t0, t1)
ax3.set_title("Band-pass anomalies with storm windows shaded")
ax3.set_xlabel("Time")
ax3.set_ylabel("z-score")
# ax3.grid(True, alpha=0.3)
ax3.legend(loc="upper right", ncol=3, fontsize=9)
common_ylim = (-3, 4)  # for example, covers most of your z-score variation
ax1.set_ylim(common_ylim)
ax2.set_ylim(common_ylim)
ax3.set_ylim(common_ylim)
plt.tight_layout()
fig_path = outdir / "wind_tke_one_figure.png"
plt.savefig(fig_path, dpi=300)
# plt.show()

print(f"Saved figure: {fig_path}")
print(f"Saved table : {summary_csv}")

#%% How does this impact mixing
# ---------- SALINITY / STRATIFICATION PREP ----------
# 1) Get ΔS (surface - bottom); if already provided, use it
if "salinity_delta_surface_minus_bottom" in ds:
    dS_raw = to_hourly(ds, "salinity_delta_surface_minus_bottom")
else:
    Ssurf = to_hourly(ds, "salinity_surface")
    Sbot  = to_hourly(ds, "salinity_bottom")
    # Align before subtracting
    dS_raw = pd.concat([Ssurf.rename("Ssurf"), Sbot.rename("Sbot")], axis=1).dropna()
    dS_raw = (dS_raw["Ssurf"] - dS_raw["Sbot"]).rename("dS")

# 2) Band-pass and z-score ΔS, then flip sign so "up = more mixed"
dS_bp = butter_bandpass(dS_raw, low_h=BP_LOW_H, high_h=BP_HIGH_H)
strat_z = -zscore(dS_bp)   # call it "strat_z": higher = LESS stratified (more mixed)

# 3) Align with wind_z / tke_z master df
df_z = pd.concat([
    wind_z.rename("wind_z"),
    tke_z.rename("tke_z"),
    strat_z.rename("strat_z")
], axis=1).dropna()

# Convenience slices
wind_z_all  = df_z["wind_z"]
tke_z_all   = df_z["tke_z"]
strat_z_all = df_z["strat_z"]

wind_z_storm  = wind_z_all[storm_mask]
tke_z_storm   = tke_z_all[storm_mask]
strat_z_storm = strat_z_all[storm_mask]

# ---------- EXTRA STATS (FULL, STORM, PER STORM) ----------
def summarize_three(name, wz, tz, sz):
    # wind↔TKE
    rWT, lagWT, pWT, slopeWT = best_lag_corr(wz, tz, max_lag_h=MAX_LAG_H)
    # wind↔(−ΔS)  (i.e., “more mixed” proxy)
    rWS, lagWS, pWS, slopeWS = best_lag_corr(wz, sz, max_lag_h=MAX_LAG_H)
    # TKE↔(−ΔS)
    rTS, lagTS, pTS, slopeTS = best_lag_corr(tz, sz, max_lag_h=MAX_LAG_H)
    return {
        "Block": name,
        "N": int(pd.concat([wz, tz, sz], axis=1).dropna().shape[0]),
        "r_best_wind_TKE": rWT, "lag_wind_TKE_h": lagWT, "p_wind_TKE": pWT,
        "r_best_wind_mixing(-dS)": rWS, "lag_wind_mixing_h": lagWS, "p_wind_mixing": pWS,
        "r_best_TKE_mixing(-dS)": rTS, "lag_TKE_mixing_h": lagTS, "p_TKE_mixing": pTS,
    }

summary_rows = []
summary_rows.append(summarize_three("Full period", wind_z_all, tke_z_all, strat_z_all))
summary_rows.append(summarize_three("Storm periods (discrete)", wind_z_storm, tke_z_storm, strat_z_storm))

# Per-storm (discrete windows, no buffers)
for letter, S, E in storm_windows:
    sel = (df_z.index >= S) & (df_z.index < E)
    summary_rows.append(
        summarize_three(f"Storm {letter} [{S:%Y-%m-%d} to {E:%Y-%m-%d}]",
                        wind_z_all[sel], tke_z_all[sel], strat_z_all[sel])
    )

summary_df2 = pd.DataFrame(summary_rows)
summary_csv2 = outdir / "wind_tke_salinity_summary_stats.csv"
summary_df2.to_csv(summary_csv2, index=False)

# ---------- UPDATE FIGURE: add TKE vs (−ΔS) scatter; add (−ΔS) to time series ----------
# (A) extra scatter (bottom-left): TKE vs mixing proxy (−ΔS) for storm periods at best lag
# Find a sensible lag using storm periods
rTS, lagTS, pTS, slopeTS = best_lag_corr(tke_z_storm, strat_z_storm, max_lag_h=MAX_LAG_H)

def lag_pair(x, y, lag_h):
    # positive lag means x leads y by lag_h hours
    if np.isnan(lag_h): 
        pair = pd.concat([x, y], axis=1).dropna()
    else:
        lag_h = int(lag_h)
        if lag_h > 0:
            pair = pd.concat([x, y.shift(-lag_h)], axis=1).dropna()
        elif lag_h < 0:
            pair = pd.concat([x.shift(+abs(lag_h)), y], axis=1).dropna()
        else:
            pair = pd.concat([x, y], axis=1).dropna()
    pair.columns = ["x", "y"]
    return pair

pair_tke_strat = lag_pair(tke_z_storm, strat_z_storm, lagTS)

# Rebuild the layout: 2x2 where bottom-left is new scatter and bottom-right is the time series
fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])

ax1 = fig.add_subplot(gs[0,0])  # scatter full (wind vs TKE)
ax2 = fig.add_subplot(gs[0,1])  # scatter storm-only (wind vs TKE)
ax3 = fig.add_subplot(gs[1,0])  # NEW: storm-only TKE vs (-ΔS)
ax4 = fig.add_subplot(gs[1,1])  # time series

# (1) Scatter: Full period (wind vs TKE, best lag)
if len(pair_full) >= 10:
    ax1.scatter(pair_full["wind_z"], pair_full["tke_z"], s=10, alpha=0.5)
    m, b = np.polyfit(pair_full["wind_z"], pair_full["tke_z"], 1)
    xx = np.linspace(pair_full["wind_z"].min(), pair_full["wind_z"].max(), 100)
    ax1.plot(xx, m*xx + b)
ax1.set_title(f"Full period (best lag = {full_stats['lag_best_h']} h, r = {full_stats['r_best']:.2f})")
ax1.set_xlabel("Wind (z, 6–72 h)"); ax1.set_ylabel("TKE (z, 6–72 h)")
ax1.grid(True, alpha=0.3)

# (2) Scatter: Storm-only (wind vs TKE, best lag)
if len(pair_storm) >= 10:
    ax2.scatter(pair_storm["wind_z"], pair_storm["tke_z"], s=10, alpha=0.5)
    m2, b2 = np.polyfit(pair_storm["wind_z"], pair_storm["tke_z"], 1)
    xx2 = np.linspace(pair_storm["wind_z"].min(), pair_storm["wind_z"].max(), 100)
    ax2.plot(xx2, m2*xx2 + b2)
ax2.set_title(f"Storm periods (best lag = {storm_stats['lag_best_h']} h, r = {storm_stats['r_best']:.2f})")
ax2.set_xlabel("Wind (z, 6–72 h)"); ax2.set_ylabel("TKE (z, 6–72 h)")
ax2.grid(True, alpha=0.3)

# (3) NEW Scatter: Storm-only TKE vs (-ΔS) mixing proxy (best lag from storms)
if len(pair_tke_strat) >= 10:
    ax3.scatter(pair_tke_strat["x"], pair_tke_strat["y"], s=10, alpha=0.5)
    m3, b3 = np.polyfit(pair_tke_strat["x"], pair_tke_strat["y"], 1)
    xx3 = np.linspace(pair_tke_strat["x"].min(), pair_tke_strat["x"].max(), 100)
    ax3.plot(xx3, m3*xx3 + b3)
ax3.set_title(f"Storm periods: TKE vs (−ΔS) (best lag = {lagTS} h, r = {rTS:.2f})")
ax3.set_xlabel("TKE (z, 6–72 h)"); ax3.set_ylabel("Mixing proxy (−ΔS, z, 6–72 h)")
ax3.grid(True, alpha=0.3)

# (4) Time series with storms shaded (wind_z, tke_z, and mixing proxy)
ax4.plot(wind_z_all.index, wind_z_all, label="Wind (z, 6–72 h)")
ax4.plot(tke_z_all.index,  tke_z_all,  label="TKE (z, 6–72 h)")
ax4.plot(strat_z_all.index, strat_z_all, label="Mixing proxy (−ΔS, z, 6–72 h)")
for i, (lab, S, E) in enumerate(storm_windows):
    ax4.axvspan(S, E, facecolor='0.85', alpha=0.20, edgecolor='0.5', linewidth=0.6,
                label="Defined storm periods" if i == 0 else None)
ax4.set_xlim(t0, t1)
ax4.set_title("Band-pass anomalies with defined storm periods shaded")
ax4.set_xlabel("Time"); ax4.set_ylabel("z-score")
ax4.legend(loc="upper right", ncol=3, fontsize=9)
ax4.grid(True, alpha=0.3)

# (optional) unify y-limits across panels (all are z-scores)
common_ylim = (-3, 4)
for ax in (ax1, ax2, ax3, ax4):
    ax.set_ylim(common_ylim)

plt.tight_layout()
fig_path2 = outdir / "wind_tke_salinity_one_figure.png"
plt.savefig(fig_path2, dpi=300)

print(f"Saved figure: {fig_path2}")
print(f"Saved table : {summary_csv2}")

#%%

# ---------- colours ----------
wind_c  = "#1f77b4"  # blue
tke_c   = "#ff7f0e"  # orange
mix_c   = "#2ca02c"  # green  (mixing proxy = -ΔS)

# Storm subsets in z-space
w_st = wind_z_all[storm_mask]
t_st = tke_z_all[storm_mask]
m_st = strat_z_all[storm_mask]   # strat_z_all = -zscore(ΔS_bandpass)

# Best-lag correlations on storms (positive lag = first variable leads)
MAX_LAG_H = 24
r_wt, lag_wt, p_wt, _ = best_lag_corr(w_st, t_st, MAX_LAG_H)   # Wind → TKE
r_wm, lag_wm, p_wm, _ = best_lag_corr(w_st, m_st, MAX_LAG_H)   # Wind → -ΔS
r_tm, lag_tm, p_tm, _ = best_lag_corr(t_st, m_st, MAX_LAG_H)   # TKE  → -ΔS
def pair_at_lag(x, y, lag_h):
    if np.isnan(lag_h): return pd.concat([x, y], axis=1).dropna()
    lag_h = int(lag_h)
    if lag_h > 0:
        return pd.concat([x, y.shift(-lag_h)], axis=1).dropna()
    elif lag_h < 0:
        return pd.concat([x.shift(+abs(lag_h)), y], axis=1).dropna()
    else:
        return pd.concat([x, y], axis=1).dropna()
# Build paired data at best lag for the scatters
pair_wt = pair_at_lag(w_st, t_st, lag_wt); pair_wt.columns = ["Wind (z)", "TKE (z)"]
pair_wm = pair_at_lag(w_st, m_st, lag_wm); pair_wm.columns = ["Wind (z)", "Mixing (−ΔS, z)"]
pair_tm = pair_at_lag(t_st, m_st, lag_tm); pair_tm.columns = ["TKE (z)",  "Mixing (−ΔS, z)"]

# ---------- FIGURE: 3 across the top, 1 long across the bottom ----------
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(13, 9))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2])

ax1 = fig.add_subplot(gs[0,0])   # Wind vs TKE (storms)
ax2 = fig.add_subplot(gs[0,1])   # Wind vs -ΔS (storms)
ax3 = fig.add_subplot(gs[0,2])   # TKE  vs -ΔS (storms)
ax4 = fig.add_subplot(gs[1,:])   # Time series (z-scores), storms shaded

# (A) Wind–TKE (positive expected)
ax1.scatter(pair_wt["Wind (z)"], pair_wt["TKE (z)"], s=14, alpha=0.55, color='black')
m,b = np.polyfit(pair_wt["Wind (z)"], pair_wt["TKE (z)"], 1)
xx = np.linspace(pair_wt["Wind (z)"].min(), pair_wt["Wind (z)"].max(), 100)
ax1.plot(xx, m*xx + b, color='black')
ax1.set_xlabel("Wind (z, 6–72 h)", color='black')
ax1.set_ylabel("TKE (z, 6–72 h)", color='black')
ax1.set_title(f"Storms: Wind vs TKE  (best lag = {lag_wt:+d} h, r = {r_wt:.2f}, p={p_wt:.3g})")
ax1.grid(True, alpha=0.3)

# (B) Wind–(−ΔS)  (positive expected)
ax2.scatter(pair_wm["Wind (z)"], pair_wm["Mixing (−ΔS, z)"], s=14, alpha=0.55, color='black')
m,b = np.polyfit(pair_wm["Wind (z)"], pair_wm["Mixing (−ΔS, z)"], 1)
xx = np.linspace(pair_wm["Wind (z)"].min(), pair_wm["Wind (z)"].max(), 100)
ax2.plot(xx, m*xx + b, color='black')
ax2.set_xlabel("Wind (z, 6–72 h)", color='black')
ax2.set_ylabel("Mixing proxy (−ΔS, z, 6–72 h)", color='black')
ax2.set_title(f"Storms: Wind vs (−ΔS)  (best lag = {lag_wm:+d} h, r = {r_wm:.2f}, p={p_wm:.3g})")
ax2.grid(True, alpha=0.3)

# (C) TKE–(−ΔS)  (positive expected)
ax3.scatter(pair_tm["TKE (z)"], pair_tm["Mixing (−ΔS, z)"], s=14, alpha=0.55, color='black')
m,b = np.polyfit(pair_tm["TKE (z)"], pair_tm["Mixing (−ΔS, z)"], 1)
xx = np.linspace(pair_tm["TKE (z)"].min(), pair_tm["TKE (z)"].max(), 100)
ax3.plot(xx, m*xx + b, color='black')
ax3.set_xlabel("TKE (z, 6–72 h)", color='black')
ax3.set_ylabel("Mixing proxy (−ΔS, z, 6–72 h)", color='black')
# ax3.set_title(f"Storms: TKE vs (−ΔS)  (best lag = {lag_tm:+d} h, r = {r_tm:.2f}, p={p_tm:.3g})")
ax3.grid(True, alpha=0.3)

# (D) Time series (z-scores) — all lines **increase with mixing**
ax4.plot(strat_z_all.index, strat_z_all, label="Mixing (−ΔS, z)",  color='deepskyblue')
ax4.plot(wind_z_all.index,  wind_z_all,  label="Wind (z, 6–72 h)", color='red', alpha = 0.5)
# ax4.plot(tke_z_all.index,   tke_z_all,   label="TKE (z, 6–72 h)",  color=tke_c)
for i, (lab, S, E) in enumerate(storm_windows):
    ax4.axvspan(S, E, facecolor='0.85', alpha=0.350, edgecolor='0.5', linewidth=0.6,
                label="Defined storm periods" if i == 0 else None)
ax4.set_title("Band-pass anomalies (z) with defined storm periods shaded")
ax4.set_xlabel("Time"); ax4.set_ylabel("z-score")
ax4.legend(loc="upper right", ncol=2, fontsize=9)
ax4.grid(True, alpha=0.3)
ax1.set_title("Storms: Wind vs TKE")
ax2.set_title("Storms: Wind vs (−ΔS)")
ax3.set_title("Storms: TKE vs (−ΔS)")
# Unify y-limits across all panels (all are z-scores)
for ax in (ax1, ax2, ax3, ax4):
    ax.set_ylim(-3, 4)
ax4.set_xlim([wind_z_all.index[0], wind_z_all.index[-1]])
plt.tight_layout()
fig_path = outdir / "wind_tke_minusdS_3top_1bottom.png"
plt.savefig(fig_path, dpi=300)
print(f"Saved figure: {fig_path}")

# ---------- CSV for z-score stats (storms only) ----------
stats_rows = []

def save_stats(name, x, y):
    r, lag, p, _ = best_lag_corr(x, y, MAX_LAG_H)
    stats_rows.append({
        "Pair": name,
        "Best lag (h)": lag,
        "Correlation (r)": r,
        "p-value": p,
        "N": int(pd.concat([x, y], axis=1).dropna().shape[0])
    })
    return r, lag, p

# Save stats for the three storm-period relationships
r_wt, lag_wt, p_wt = save_stats("Wind vs TKE",       w_st, t_st)
r_wm, lag_wm, p_wm = save_stats("Wind vs −ΔS",       w_st, m_st)
r_tm, lag_tm, p_tm = save_stats("TKE vs −ΔS",        t_st, m_st)

# Write the CSV
stats_df = pd.DataFrame(stats_rows)
stats_csv = outdir / "all_zscore_stats.csv"
stats_df.to_csv(stats_csv, index=False)
print(f"Saved stats CSV: {stats_csv}")
