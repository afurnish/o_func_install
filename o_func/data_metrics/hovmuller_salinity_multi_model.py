#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create multi-panel salinity Hovmöller plots:
A: IRENE (climatology)
B: IRENE (real rivers)
C: UKC4
D: Δ(IRENE_clim − UKC4)
E: Δ(IRENE_real − UKC4)
F: Inset map (right column; same sizing)
"""

import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import gridspec
import cartopy.crs as ccrs
import geopandas as gpd

from o_func import opsys
start_path = Path(opsys('PNC'))

# -------- X-tick control (adjust just these) --------
XTICK_EVERY_DAYS = 7          # <--- change to 3, 5, 14, etc.
XTICK_DATE_FMT   = '%Y-%m-%d' # label format
USE_MINOR_DAILY_TICKS = True  # set False to disable minor (daily) ticks

# --------------------------- Paths ---------------------------
pkl_clim = start_path / 'modelling_DATA/kent_estuary_project/13.3D_finals/models/3d_10_layer_climatology_layer0/outputs/data_proc/salinity_hovmuller_data.pkl'
pkl_real = start_path / 'modelling_DATA/kent_estuary_project/13.3D_finals/models/3d_10_layer_realriv_layer0/outputs/data_proc/salinity_hovmuller_data.pkl'
storm_meta_name = 'skew_storm_surge_saved.pkl'
outline_path = start_path / "modelling_DATA/kent_estuary_project/5.Final/QGIS/kent_area_poly_as_lines.shp"
save_dir = start_path / "modelling_DATA/kent_estuary_project/salinity_plots"

# --------------------------- Helpers ---------------------------

def _edges_from_centers(x):
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return np.array([x[0]-0.5, x[0]+0.5])
    dx = np.diff(x)
    left  = x[0]  - dx[0]/2
    right = x[-1] + dx[-1]/2
    mids = (x[:-1] + x[1:]) / 2
    return np.r_[left, mids, right]

def _pad_to_square(bounds, pad_frac=0.10):
    lon0, lon1, lat0, lat1 = bounds
    w = lon1 - lon0
    h = lat1 - lat0
    size = max(w, h)
    cx = 0.5 * (lon0 + lon1); cy = 0.5 * (lat0 + lat1)
    half = 0.5 * size * (1 + pad_frac)
    return (cx - half, cx + half, cy - half, cy + half)

def _assert_same_time_and_distance(d1, d2, est):
    t1 = np.asarray(d1['time']); t2 = np.asarray(d2['time'])
    x1 = np.asarray(d1['distance']); x2 = np.asarray(d2['distance'])
    if t1.shape != t2.shape or np.any(t1 != t2):
        raise ValueError(f"[{est}] time arrays differ between pickles.")
    if x1.shape != x2.shape or np.any(x1 != x2):
        raise ValueError(f"[{est}] distance arrays differ between pickles.")
    return t1, x1

def _strict_mask_three(SA1, SA2, SB):
    g1 = np.isfinite(SA1).all(axis=0)
    g2 = np.isfinite(SA2).all(axis=0)
    gB = np.isfinite(SB).all(axis=0)
    return g1 & g2 & gB

def _norm_est_key_for_meta_lower_to_caps(est_name_lower):
    return est_name_lower.capitalize()

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_storm_meta_from_dir(data_proc_dir: Path):
    meta_path = data_proc_dir / storm_meta_name
    if not meta_path.exists():
        raise FileNotFoundError(f"Storm meta pickle not found: {meta_path}")
    raw = load_pickle(meta_path)
    estuary_bounds, store_td_valid, store_td_invalid = {}, {}, {}
    for est_caps, payload in raw.items():
        if not isinstance(payload, dict):
            continue
        if 'est_bounds' in payload: estuary_bounds[est_caps] = payload['est_bounds']
        if 'td_valid'   in payload: store_td_valid[est_caps] = payload['td_valid']
        if 'td_invalid' in payload: store_td_invalid[est_caps] = payload['td_invalid']
    return estuary_bounds, store_td_valid, store_td_invalid

from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox, VPacker, TextArea, DrawingArea
import matplotlib.patches as mpatches

def add_north_arrow(ax, size=0.08, location=(0.1, 0.9), pad=0.02):
    """
    Add a simple north arrow to a Cartopy GeoAxes.
    size: relative arrow size (fraction of map extent)
    location: (x,y) axes fraction for arrow tip
    """
    arrow = mpatches.FancyArrow(0, 0, 0, size,
                                width=size * 0.3, head_width=size * 0.5,
                                head_length=size * 0.5, facecolor='k', edgecolor='k')
    box = AuxTransformBox(ax.transAxes)
    box.add_artist(arrow)

    text = TextArea("N", textprops=dict(color='k', fontsize=10, weight='bold'))
    packed = VPacker(children=[text, box], align="center", pad=0, sep=2)
    anchored = AnchoredOffsetbox(loc='center',
                                 child=packed, pad=pad,
                                 frameon=False,
                                 bbox_to_anchor=location,
                                 bbox_transform=ax.transAxes,
                                 borderpad=0.)
    ax.add_artist(anchored)
# --------------------------- Plotting ---------------------------

def hovmoller_salinity_multi(
    est_name,                 # lower-case key for salinity bundles
    data_irene_clim,
    data_irene_real,
    estuary_bounds,
    store_td_valid,
    store_td_invalid,
    outline_gdf,
    outdir,
    sal_vlim=(0, 35)
):
    # --- pull arrays ---
    if est_name not in data_irene_clim or est_name not in data_irene_real:
        print(f"[{est_name}] Not in both pickles; skipping."); return

    time, distances_all = _assert_same_time_and_distance(
        data_irene_clim[est_name], data_irene_real[est_name], est_name
    )
    idx_points = np.asarray(data_irene_clim[est_name]['idx_points'], dtype=int)

    SA_clim = np.asarray(data_irene_clim[est_name]['S_A'])[:, idx_points]
    SB_ref  = np.asarray(data_irene_clim[est_name]['S_B'])[:, idx_points]
    SA_real = np.asarray(data_irene_real[est_name]['S_A'])[:, idx_points]
    d_m     = np.asarray(distances_all)[idx_points]

    good_cols = _strict_mask_three(SA_clim, SA_real, SB_ref)
    if good_cols.sum() == 0:
        print(f"[{est_name}] No common NaN-free columns across IRENEs and UKC4; skipping."); return

    d_m, SA_c, SA_r, SB = d_m[good_cols], SA_clim[:, good_cols], SA_real[:, good_cols], SB_ref[:, good_cols]
    order = np.argsort(d_m)
    d_m, SA_c, SA_r, SB = d_m[order], SA_c[:, order], SA_r[:, order], SB[:, order]

    t_nums  = mdates.date2num(np.asarray(time))
    t_edges = _edges_from_centers(t_nums)
    d_edges = _edges_from_centers(d_m / 1000.0)  # km

    sal_min = float(np.nanmin([SA_c.min(), SA_r.min(), SB.min()]))
    sal_max = float(np.nanmax([SA_c.max(), SA_r.max(), SB.max()]))

    DIFF_c = SA_c - SB
    DIFF_r = SA_r - SB

    # Per-estuary symmetric Δ limits (centered at 0)
    local_max = float(np.nanmax([np.nanmax(np.abs(DIFF_c)), np.nanmax(np.abs(DIFF_r))]))
    if not np.isfinite(local_max) or local_max == 0: local_max = 1.0
    vmin_diff, vmax_diff = -local_max, local_max

    # --- figure with RIGHT inset ---

    # Increase total figure height (e.g., 14.0 or 16.0 inches)
    fig = plt.figure(figsize=(13.5, 20.0))
    
    # Make left panels taller but keep right inset fixed height
    gs = gridspec.GridSpec(
        nrows=5, ncols=2,
        width_ratios=[1.0, 0.42],
        height_ratios=[1.8, 1.8, 1.8, 1.8, 1.8],  # Taller rows for left panels
        wspace=0.15, hspace=0.3
    )


    axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[1, 0], sharex=axA, sharey=axA)
    axC = fig.add_subplot(gs[2, 0], sharex=axA, sharey=axA); axD = fig.add_subplot(gs[3, 0], sharex=axA, sharey=axA)
    axE = fig.add_subplot(gs[4, 0], sharex=axA, sharey=axA)
    axes_left = [axA, axB, axC, axD, axE]

    cm, cmd = plt.get_cmap('viridis'), plt.get_cmap('RdBu_r')

    imA = axA.pcolormesh(t_edges, d_edges, SA_c.T, cmap=cm, shading='flat',
                         edgecolors='none', vmin=sal_min, vmax=sal_max)
    axA.set_title(f"A ({est_name.capitalize()} - IRENE$_{{C10}}$)"); axA.set_ylabel("Distance [km]")
    fig.colorbar(imA, ax=axA, label="Salinity [psu]", pad=0.012, fraction=0.046)

    imB = axB.pcolormesh(t_edges, d_edges, SA_r.T, cmap=cm, shading='flat',
                         edgecolors='none', vmin=sal_min, vmax=sal_max)
    axB.set_title(f"B ({est_name.capitalize()} - IRENE$_{{R10}}$)"); axB.set_ylabel("Distance [km]")
    fig.colorbar(imB, ax=axB, label="Salinity [psu]", pad=0.012, fraction=0.046)

    imC = axC.pcolormesh(t_edges, d_edges, SB.T, cmap=cm, shading='flat',
                         edgecolors='none', vmin=sal_min, vmax=sal_max)
    axC.set_title(f"C ({est_name.capitalize()} — UKC4)"); axC.set_ylabel("Distance [km]")
    fig.colorbar(imC, ax=axC, label="Salinity [psu]", pad=0.012, fraction=0.046)

    imD = axD.pcolormesh(t_edges, d_edges, DIFF_c.T, cmap=cmd, shading='flat',
                         edgecolors='none', vmin=vmin_diff, vmax=vmax_diff)
    axD.set_title(f"D ({est_name.capitalize()} — ΔIRENE$_{{C10}}$)"); axD.set_ylabel("Distance [km]")
    fig.colorbar(imD, ax=axD, label="ΔSalinity [psu]", pad=0.012, fraction=0.046)

    imE = axE.pcolormesh(t_edges, d_edges, DIFF_r.T, cmap=cmd, shading='flat',
                         edgecolors='none', vmin=vmin_diff, vmax=vmax_diff)
    axE.set_title(f"E ({est_name.capitalize()} — ΔIRENE$_{{R10}}$)"); axE.set_ylabel("Distance [km]")
    axE.set_xlabel("Time")
    fig.colorbar(imE, ax=axE, label="ΔSalinity [psu]", pad=0.012, fraction=0.046)

    # ---------- EVENLY-SPACED X TICKS (shared DayLocator) ----------
    major_locator = mdates.DayLocator(interval=XTICK_EVERY_DAYS)
    major_fmt     = mdates.DateFormatter(XTICK_DATE_FMT)
    minor_locator = mdates.DayLocator(interval=1) if USE_MINOR_DAILY_TICKS else None

    for ax in axes_left:
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_formatter(major_fmt)
        if minor_locator is not None:
            ax.xaxis.set_minor_locator(minor_locator)

    for ax in axes_left[:-1]:
        ax.tick_params(axis='x', which='major', bottom=True, labelbottom=False,
                       direction='out', length=3)
        if minor_locator is not None:
            ax.tick_params(axis='x', which='minor', bottom=True, labelbottom=False,
                           direction='out', length=2)

    axes_left[-1].tick_params(axis='x', which='major', bottom=True, labelbottom=True,
                              direction='out', length=3)
    if minor_locator is not None:
        axes_left[-1].tick_params(axis='x', which='minor', bottom=True, labelbottom=False,
                                  direction='out', length=2)

    fig.autofmt_xdate()

    # Clamp salinity to (0,35)
    v0, v1 = sal_vlim
    imA.set_clim(v0, v1); imB.set_clim(v0, v1); imC.set_clim(v0, v1)

    # ---- RIGHT COLUMN INSET ----
    est_caps = _norm_est_key_for_meta_lower_to_caps(est_name)
    est_bounds = estuary_bounds[est_caps]
    td_valid   = store_td_valid[est_caps]
    td_invalid = store_td_invalid[est_caps]

    ax_map = fig.add_subplot(gs[:, 1], projection=ccrs.PlateCarree())
    ax_map.set_facecolor('white'); ax_map.set_title('F')

    # arrow_x, arrow_y = 0.1, 0.95  # top-left corner; adjust as needed

    # ax_map.annotate(
    #     'N',
    #     xy=(arrow_x, arrow_y), xytext=(arrow_x, arrow_y - 0.08),
    #     xycoords='axes fraction', textcoords='axes fraction',
    #     ha='center', va='center',
    #     fontsize=10, fontweight='bold',
    #     arrowprops=dict(facecolor='black', width=2, headwidth=8, headlength=10)
    # )
    add_north_arrow(ax_map, size=0.08, location=(0.1, 0.9))
    
    outline_gdf.plot(ax=ax_map, edgecolor='black', facecolor='none',
                     linewidth=0.8, transform=ccrs.PlateCarree())

    def _xy(obj):
        try: return obj['x'], obj['y']
        except Exception: return None, None
    xv, yv = _xy(td_valid); xi, yi = _xy(td_invalid)
    if xv is not None and yv is not None and len(xv):
        ax_map.scatter(xv, yv, s=12, color='red', transform=ccrs.PlateCarree(), zorder=3)
    if xi is not None and yi is not None and len(xi):
        ax_map.scatter(xi, yi, s=12, color='green', transform=ccrs.PlateCarree(), zorder=3)

    lon0, lon1, lat0, lat1 = _pad_to_square(est_bounds, pad_frac=0.10)
    ax_map.set_extent((lon0, lon1, lat0, lat1), crs=ccrs.PlateCarree())
    ax_map.set_xticks([]); ax_map.set_yticks([])
    for spine in ax_map.spines.values():
        spine.set_visible(True); spine.set_linewidth(0.8)

    # Save
    outdir.mkdir(parents=True, exist_ok=True)
    savename = outdir / f"{est_name}_hovmoller_multi_irene_vs_ukc4_with_inset.png"
    fig.savefig(savename, dpi=300, bbox_inches='tight')
    # plt.close(fig)

    print(f"[{est_name}] plotted {d_m.size} common clean transect points "
          f"[d range {d_m.min():.0f}–{d_m.max():.0f} m]. Saved to {savename}")

# --------------------------- Run ---------------------------

if __name__ == "__main__":
    data_clim = load_pickle(pkl_clim)
    data_real = load_pickle(pkl_real)

    clim_dir = pkl_clim.parent; real_dir = pkl_real.parent
    estuary_bounds_c, store_td_valid_c, store_td_invalid_c = load_storm_meta_from_dir(clim_dir)
    estuary_bounds_r, store_td_valid_r, store_td_invalid_r = load_storm_meta_from_dir(real_dir)

    estuary_bounds  = {**estuary_bounds_c,  **estuary_bounds_r}
    store_td_valid  = {**store_td_valid_c,  **store_td_valid_r}
    store_td_invalid= {**store_td_invalid_c,**store_td_invalid_r}

    outline_gdf = gpd.read_file(outline_path)

    common_ests = sorted(set(data_clim.keys()) & set(data_real.keys()))
    for est in common_ests:
        if not isinstance(data_clim.get(est), dict) or 'S_A' not in data_clim.get(est, {}):
            continue
        hovmoller_salinity_multi(
            est_name=est,
            data_irene_clim=data_clim,
            data_irene_real=data_real,
            estuary_bounds=estuary_bounds,
            store_td_valid=store_td_valid,
            store_td_invalid=store_td_invalid,
            outline_gdf=outline_gdf,
            outdir=save_dir,
            sal_vlim=(0, 35)
        )
