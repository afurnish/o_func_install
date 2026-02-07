#!/usr/bin/env python3
"""
Combine skew-storm-surge Hovmöllers (climatology vs real-river) with wind and a common inset map.
Optionally include a bottom panel showing the skew-surge difference (real − climatology).

Inputs (edit the PATHS section below):
- CLIM_PKL: pickle produced by hovmoler_skew_surge_plot_with_wind() for the climatology run
- REAL_PKL: pickle produced by hovmoler_skew_surge_plot_with_wind() for the real-river run

Outputs:
- One PNG per estuary in OUT_DIR named:
  "{estuary}_skew_surge_hovmoller_combo.png"  (or "..._with_diff.png" if difference panel is enabled)
"""

from __future__ import annotations
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import gridspec
import cartopy.crs as ccrs  # required to render the map inset

from o_func import opsys
start_path = Path(opsys('PNC'))

# --------------------------
# PATHS (edit as needed)
# --------------------------
CLIM_PKL = start_path / Path("modelling_DATA/kent_estuary_project/13.3D_finals/models/3d_10_layer_climatology_layer0/outputs/data_proc/skew_storm_surge_saved.pkl")
REAL_PKL = start_path / Path("modelling_DATA/kent_estuary_project/13.3D_finals/models/3d_10_layer_realriv_layer0/outputs/data_proc/skew_storm_surge_saved.pkl")
fig_path = start_path / 'modelling_DATA/kent_estuary_project/storm_surge'
OUT_DIR  = fig_path / "skew_surge_combo_figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
# --------------------------
# CONFIG
# --------------------------

# --------------------------
# Helpers
# --------------------------
def _edges_from_centers(centers: np.ndarray) -> np.ndarray:
    """Compute bin edges from center coordinates for pcolormesh."""
    centers = np.asarray(centers, dtype=float)
    if centers.size == 1:
        c = centers[0]
        return np.array([c - 0.5, c + 0.5], dtype=float)
    d = np.diff(centers)
    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = centers[:-1] + 0.5 * d
    edges[0] = centers[0] - 0.5 * d[0]
    edges[-1] = centers[-1] + 0.5 * d[-1]
    return edges

def _pad_to_square(bounds, pad_frac: float = 0.10):
    """
    Pad (lon_min, lon_max, lat_min, lat_max) to a square extent with margin.
    Returns (lon0, lon1, lat0, lat1).
    """
    lon0, lon1, lat0, lat1 = bounds
    w = lon1 - lon0
    h = lat1 - lat0
    side = max(w, h)
    cx = (lon0 + lon1) * 0.5
    cy = (lat0 + lat1) * 0.5
    half = 0.5 * side * (1.0 + pad_frac)
    return (cx - half, cx + half, cy - half, cy + half)

def _first_nonempty_df(*dfs):
    """Return the first DataFrame that is not None and not empty, else None."""
    for df in dfs:
        if df is not None and not getattr(df, "empty", True):
            return df
    return None

def _safe_get(d: dict, k: str, default=None):
    return d[k] if k in d else default

# --------------------------
# Main plotting function
# --------------------------
def make_skew_surge_hovmoller_combo(
    clim_pkl: Path,
    real_pkl: Path,
    out_dir: Path,
    add_difference: bool = False
):
    with open(clim_pkl, "rb") as f:
        clim = pickle.load(f)
    with open(real_pkl, "rb") as f:
        real = pickle.load(f)

    estuaries = sorted(set(clim.keys()).intersection(set(real.keys())))
    if not estuaries:
        raise RuntimeError("No overlapping estuary keys in the two pickle files.")

    for est in estuaries:
        dA = clim[est]  # climatology dict for this estuary
        dB = real[est]  # real-river   dict for this estuary

        # ---- Core arrays ----
        time_A = pd.to_datetime(dA["time_vector"])
        time_B = pd.to_datetime(dB["time_vector"])
        tmin = min(time_A.min(), time_B.min())
        tmax = max(time_A.max(), time_B.max())

        tA_nums  = mdates.date2num(time_A.values)
        tB_nums  = mdates.date2num(time_B.values)
        tA_edges = _edges_from_centers(tA_nums)
        tB_edges = _edges_from_centers(tB_nums)

        dist_A_m = np.asarray(dA["dist_subset_cleaned"], dtype=float)
        dist_B_m = np.asarray(dB["dist_subset_cleaned"], dtype=float)
        dist_A_km = dist_A_m
        dist_B_km = dist_B_m 
        dA_edges = _edges_from_centers(dist_A_km)
        dB_edges = _edges_from_centers(dist_B_km)

        data_A = np.asarray(dA["data_cleaned"], dtype=float)  # shape [ny, nt]
        data_B = np.asarray(dB["data_cleaned"], dtype=float)

        # Symmetric vmin/vmax across BOTH runs for fair comparison
        vmax_abs = np.nanmax([np.nanmax(np.abs(data_A)), np.nanmax(np.abs(data_B))])
        vmin = -vmax_abs
        vmax =  vmax_abs

        # Difference (real − climatology)
        if add_difference:
            # We’ll interpolate to a common (time, distance) grid only if centers differ.
            # If shapes and coordinates match, just subtract along the intersection.
            # For simplicity/robustness: use nearest common subset when not identical.
            # (Most of your transect exports are aligned, so this will usually match directly.)
            diff = None
            if data_A.shape == data_B.shape and np.allclose(dist_A_km, dist_B_km) and len(tA_nums) == len(tB_nums) and np.allclose(tA_nums, tB_nums):
                diff = data_B - data_A
                tD_edges = tA_edges
                dD_edges = dA_edges
            else:
                # Align on overlapping distance indices
                # (Simple nearest-neighbour approach; refine later if needed.)
                common_km = np.intersect1d(np.round(dist_A_km, 6), np.round(dist_B_km, 6))
                ia = np.nonzero(np.isin(np.round(dist_A_km, 6), common_km))[0]
                ib = np.nonzero(np.isin(np.round(dist_B_km, 6), common_km))[0]
                # Align time by intersection of timestamps
                common_t = np.intersect1d(tA_nums, tB_nums)
                ja = np.nonzero(np.isin(tA_nums, common_t))[0]
                jb = np.nonzero(np.isin(tB_nums, common_t))[0]
                if ia.size and ib.size and ja.size and jb.size:
                    diff = data_B[ib[:, None], jb] - data_A[ia[:, None], ja]
                    tD_edges = _edges_from_centers(common_t)
                    dD_edges = _edges_from_centers(common_km)
                else:
                    print(f"[{est}] Could not align grids for difference panel; skipping difference.")
                    add_diff_here = False
            if diff is not None:
                add_diff_here = True
                diff_vmax = float(np.nanmax(np.abs(diff)))
                diff_vmin = -diff_vmax
            else:
                add_diff_here = False
        else:
            add_diff_here = False

        # ---- Map / points ----
        outline    = _first_nonempty_df(_safe_get(dA, "outline"), _safe_get(dB, "outline"))
        
        est_bounds = _safe_get(dA, "est_bounds")
        if est_bounds is None:
            est_bounds = _safe_get(dB, "est_bounds")
        td_valid_A = _safe_get(dA, "td_valid")
        td_invalid_A = _safe_get(dA, "td_invalid")
        td_valid_B = _safe_get(dB, "td_valid")
        td_invalid_B = _safe_get(dB, "td_invalid")

        # ---- Wind ----
        wind_df    = _first_nonempty_df(_safe_get(dA, "wind_df"), _safe_get(dB, "wind_df"))

        # ---- Figure layout (3 columns: main | colorbar | map) ----
        nrows = 4 if add_diff_here else 3
        fig = plt.figure(figsize=(15, 9 if nrows == 3 else 11))
        gs = gridspec.GridSpec(
            nrows=nrows, ncols=3,
            # Change the theird number along for spacing. 
            width_ratios=[30, 1, 12],    # main | cbar | map
            height_ratios=([1.3, 1.3, 1.3] if nrows == 3 else [1.0, 1.2, 1.2, 1.2]),
            wspace=0.07, hspace=0.25
        )

        # Row 1: WIND (sharex with below)
        ax_wind = fig.add_subplot(gs[0, 0])
        ax_wind_cbar_dummy = fig.add_subplot(gs[0, 1])
        ax_wind_cbar_dummy.axis("off")

        if wind_df is not None:
            w = wind_df.copy()
            w["datetime"] = pd.to_datetime(w["datetime"])
            ax_wind.plot(w["datetime"], w["speed (m/s)"], label="Wind speed (m s$^{-1}$)", color = "black")
            ax_dir = ax_wind.twinx()
            ax_dir.plot(w["datetime"], w["direction (deg)"], alpha=0.7, label="Wind dir (°)", color="gray")
            ax_dir.set_ylabel("Wind direction [°]", color = 'gray')
            ax_dir.set_ylim(0, 360)

        ax_wind.set_ylabel("Wind speed [m s$^{-1}$]", color = 'black')
        ax_wind.set_xlim(tmin, tmax)
        ax_wind.tick_params(axis='x', labelbottom=False)
        ax_wind.set_title("A")

        # Row 2: CLIM Hovmöller
        axA = fig.add_subplot(gs[1, 0], sharex=ax_wind)
        pcmA = axA.pcolormesh(
            tA_edges, dA_edges, data_A,
            cmap="bwr", shading="flat", vmin=vmin, vmax=vmax
        )
        axA.set_title("B ($\Delta$IRENE$_{C1}$)")
        axA.set_ylabel("Distance along estuary [km]")
        caxA = fig.add_subplot(gs[1, 1])
        cbarA = fig.colorbar(pcmA, cax=caxA)
        cbarA.set_label("Skew surge residual [m]")
        axA.tick_params(axis='x', labelbottom=False)

        # Row 3: REAL Hovmöller
        row_real = 2
        axB = fig.add_subplot(gs[row_real, 0], sharex=ax_wind)
        pcmB = axB.pcolormesh(
            tB_edges, dB_edges, data_B,
            cmap="bwr", shading="flat", vmin=vmin, vmax=vmax
        )
        axB.set_title("C ($\Delta$IRENE$_{R1}$)")
        axB.set_ylabel("Distance along estuary [km]")
        axB.set_xlabel("Time")
        caxB = fig.add_subplot(gs[row_real, 1])
        cbarB = fig.colorbar(pcmB, cax=caxB)
        cbarB.set_label("Skew surge residual [m]")

        # Optional Row 4: DIFFERENCE (real − climatology)
        if add_diff_here:
            axD = fig.add_subplot(gs[3, 0], sharex=ax_wind)
            pcmD = axD.pcolormesh(
                tD_edges, dD_edges, diff,
                cmap="bwr", shading="flat", vmin=diff_vmin, vmax=diff_vmax
            )
            # axD.set_title("Difference (real − climatology)", loc="left")
            axD.set_ylabel("Distance along estuary [km]")
            axD.set_xlabel("Time")
            caxD = fig.add_subplot(gs[3, 1])
            cbarD = fig.colorbar(pcmD, cax=caxD)
            cbarD.set_label("Skew surge difference [m]")

        # Tidy date labels (bottom-most main panel only)
        (axD if add_diff_here else axB).xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()

        # Right column: MAP spanning all rows
        # ax_map = fig.add_subplot(gs[:, 2], projection=ccrs.PlateCarree())
        ax_map = fig.add_axes([0.74, 0.08, 0.18, 0.9], projection=ccrs.PlateCarree())
        ax_map.set_facecolor("white")

        if outline is not None:
            outline.plot(ax=ax_map, edgecolor='black', facecolor='none',
                         linewidth=0.8, transform=ccrs.PlateCarree())

        # Valid / invalid points (prefer climatology; fall back to real-river)
        if td_valid_A is not None and len(td_valid_A):
            ax_map.scatter(td_valid_A["x"], td_valid_A["y"], s=12, color="red",
                           zorder=3, transform=ccrs.PlateCarree())
        elif td_valid_B is not None and len(td_valid_B):
            ax_map.scatter(td_valid_B["x"], td_valid_B["y"], s=12, color="red",
                           zorder=3, transform=ccrs.PlateCarree())

        if td_invalid_A is not None and len(td_invalid_A):
            ax_map.scatter(td_invalid_A["x"], td_invalid_A["y"], s=12, color="green",
                           zorder=3, transform=ccrs.PlateCarree())
        elif td_invalid_B is not None and len(td_invalid_B):
            ax_map.scatter(td_invalid_B["x"], td_invalid_B["y"], s=12, color="green",
                           zorder=3, transform=ccrs.PlateCarree())

        if est_bounds is not None:
            lon0, lon1, lat0, lat1 = _pad_to_square(est_bounds, pad_frac=0.10)
            ax_map.set_extent((lon0, lon1, lat0, lat1), crs=ccrs.PlateCarree())
        ax_map.set_xticks([]); ax_map.set_yticks([])
        for spine in ax_map.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
        ax_map.set_title("D")

        # Save
        suffix = "_with_diff" if add_diff_here else ""
        savename = out_dir / f"{est}_skew_surge_hovmoller_combo{suffix}.png"
        fig.savefig(savename, dpi=300, bbox_inches="tight")
        # plt.close(fig)
        print(f"Saved: {savename}")

# --------------------------
# Run
# --------------------------
if __name__ == "__main__":
    make_skew_surge_hovmoller_combo(
        clim_pkl=CLIM_PKL,
        real_pkl=REAL_PKL,
        out_dir=OUT_DIR,
        add_difference=False,
    )


