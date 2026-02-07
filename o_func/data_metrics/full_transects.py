#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 2 — Phase-aligned estuarine transect plots at mouth-peak times,
plus statistics CSV and outline figure (no basemap) with exact UKC4 & IRENE boundaries.

This version builds the cache exactly like STEP 1b:
- trims the first 10 days of IRENE (unstructured) data
- resamples IRENE to hourly means and stamps at :30
- stores 20-min IRENE series, hourly IRENE series, nearest face indices,
  bathymetry-at-points, and point lon/lat into transect_timeseries_cache.nc

Outputs (per base in base_list):
- figures/transects_phase_aligned_MAX.png
- figures/transects_points_outline_MAX.png
- figures/transect_stats_max.csv
- figures/transect_timeseries_cache.nc
- figures/ukc4_outline_exact.shp (+ sidecar files)
- (from cache build) figures/mouth_timeseries_<EXTREME>_HAVERSINE_CUTOFF_HOURLY.png
- (from cache build) figures/mouth_peak_times_<EXTREME>_haversine_cutoff_hourly.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree  # haversine distance
import geopandas as gpd
from shapely.geometry import Polygon, LineString, MultiPolygon
from shapely.ops import unary_union, linemerge, polygonize
from matplotlib.lines import Line2D
import xugrid as xu
from o_func import opsys

# ===================== CONFIG =====================
base_list = [
    Path('/media/af/PNB_extra/scw_10layer_realriv'),
    Path('/media/af/PNB_extra/scw_10_layer_climatology'),
    Path('/media/af/PNB_extra/runSCW_ao_yawind_orig8RealRiver_m0.035_Forcing_85_Discouv-58757607'),
    Path('/media/af/PNB_extra/runSCW_ao_yawind_AllRivNoDuddonClimatology_m0.035_Forcing_85_Discouv-58757598'),
]

# Controls
EXTREME       = 'max'       # 'max' or 'min'
FIGSIZE       = (8, 12)
DPI           = 300
TRIM_FIRST_DAYS = 10        # cut first N days when building cache
FORCE_REBUILD_CACHE = False # set True to overwrite any existing cache

# Colours
CLR_IRENE = "#1f77b4"  # blue
CLR_UKC4  = "#ff7f0e"  # orange
CLR_BATHY = "#2ca02c"  # green

# Unified line width everywhere
LINE_W = 0.8
# ==================================================


# ---------- helpers ----------
def pick_first_existing(ds, names):
    for n in names:
        if n in ds.variables:
            return n
    raise KeyError(f"None of {names} found in variables.")

def pick_first_coord(ds, names):
    for n in names:
        if n in ds.coords:
            return n
    raise KeyError(f"None of {names} found in coords.")

def ensure_time_first(var, time_coord):
    return var if var.dims[0] == time_coord else var.transpose(time_coord, ...)

def deg2rad_xy(lon_deg, lat_deg):
    lon_r = np.radians(np.asarray(lon_deg).reshape(-1))
    lat_r = np.radians(np.asarray(lat_deg).reshape(-1))
    return np.column_stack((lat_r, lon_r))  # BallTree expects [lat, lon] in radians

def build_balltree_grid(lon2d, lat2d):
    pts_rad = deg2rad_xy(lon2d, lat2d).reshape(-1, 2)  # flatten to (ny*nx,2)
    return BallTree(pts_rad, metric='haversine'), lon2d.shape

def build_balltree_points(lon1d, lat1d):
    return BallTree(deg2rad_xy(lon1d, lat1d), metric='haversine')

def int_to_letter(n):
    return chr(ord('A') + int(n))

def cumulative_dist_km(xs_deg, ys_deg):
    lon = np.radians(np.asarray(xs_deg))
    lat = np.radians(np.asarray(ys_deg))
    dlon = np.diff(lon)
    dlat = np.diff(lat)
    a = np.sin(dlat/2.0)**2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon/2.0)**2
    c = 2*np.arcsin(np.sqrt(a))
    return np.insert(6371.0 * np.cumsum(c), 0, 0.0)

def find_face_dim(var, time_dim):
    other = [d for d in var.dims if d != time_dim]
    if not other:
        raise ValueError(f"Cannot infer face dim in {var.dims}")
    for d in other:
        if 'face' in d.lower():
            return d
    if len(other) == 1:
        return other[0]
    sizes = {d: var.sizes[d] for d in other}
    return max(sizes, key=sizes.get)

def nearest_time_index(arr_dt64, target_dt64):
    arr = np.asarray(arr_dt64)
    return int(np.argmin(np.abs(arr - np.datetime64(target_dt64))))

def three_20min_indices_for_hour(unstr_times, peak_halfhour):
    base = (pd.to_datetime(peak_halfhour) - pd.Timedelta(minutes=30)).floor('H')
    targets = [base, base + pd.Timedelta(minutes=20), base + pd.Timedelta(minutes=40)]
    return tuple(nearest_time_index(unstr_times, np.datetime64(t)) for t in targets)

# ---- exact outline helpers (UKC4) ----
def centers_to_vertices(Xc, Yc):
    """Build a (ny+1, nx+1) vertex grid from center lon/lat (ny, nx)."""
    ny, nx = Xc.shape
    Xv = np.empty((ny+1, nx+1), dtype=float)
    Yv = np.empty((ny+1, nx+1), dtype=float)

    # interior
    Xv[1:-1, 1:-1] = 0.25*(Xc[:-1, :-1] + Xc[:-1, 1:] + Xc[1:, :-1] + Xc[1:, 1:])
    Yv[1:-1, 1:-1] = 0.25*(Yc[:-1, :-1] + Yc[:-1, 1:] + Yc[1:, :-1] + Yc[1:, 1:])

    # top/bottom edges
    Xv[0, 1:-1]  = 0.5*(Xc[0, :-1]  + Xc[0, 1:]);    Yv[0, 1:-1]  = 0.5*(Yc[0, :-1]  + Yc[0, 1:])
    Xv[-1, 1:-1] = 0.5*(Xc[-1, :-1] + Xc[-1, 1:]);   Yv[-1, 1:-1] = 0.5*(Yc[-1, :-1] + Yc[-1, 1:])

    # left/right edges
    Xv[1:-1, 0]  = 0.5*(Xc[:-1, 0]  + Xc[1:, 0]);    Yv[1:-1, 0]  = 0.5*(Yc[:-1, 0]  + Yc[1:, 0])
    Xv[1:-1, -1] = 0.5*(Xc[:-1, -1] + Xc[1:, -1]);   Yv[1:-1, -1] = 0.5*(Yc[:-1, -1] + Yc[1:, -1])

    # corners via simple bilinear extrapolation
    Xv[0,0]   = Xv[0,1]   + Xv[1,0]   - Xv[1,1];     Yv[0,0]   = Yv[0,1]   + Yv[1,0]   - Yv[1,1]
    Xv[0,-1]  = Xv[0,-2]  + Xv[1,-1]  - Xv[1,-2];    Yv[0,-1]  = Yv[0,-2]  + Yv[1,-1]  - Yv[1,-2]
    Xv[-1,0]  = Xv[-1,1]  + Xv[-2,0]  - Xv[-2,1];    Yv[-1,0]  = Yv[-1,1]  + Yv[-2,0]  - Yv[-2,1]
    Xv[-1,-1] = Xv[-1,-2] + Xv[-2,-1] - Xv[-2,-2];   Yv[-1,-1] = Yv[-1,-2] + Yv[-2,-1] - Yv[-2,-2]
    return Xv, Yv

def outline_curvilinear_from_mask(Xc, Yc, mask):
    """Exact cell-edge outline (Multi)Polygon for a curvilinear quad grid given a boolean mask (ny, nx)."""
    Xv, Yv = centers_to_vertices(Xc, Yc)
    ny, nx = mask.shape
    polys = []
    for j in range(ny):
        for i in range(nx):
            if mask[j, i]:
                corners = [(Xv[j, i],   Yv[j, i]),
                           (Xv[j, i+1], Yv[j, i+1]),
                           (Xv[j+1,i+1],Yv[j+1,i+1]),
                           (Xv[j+1,i],  Yv[j+1,i])]
                polys.append(Polygon(corners))
    if not polys:
        return None
    return unary_union(polys)

# ---------- cache builder (EXACTLY your Step-1b logic) ----------
def build_cache_if_needed(CACHE_NC, regrid_path, unstr_path, transect_csv, bathy_file,
                          extreme='max', cutoff_days=10, force=False, dpi=300):
    """
    Build STEP-1b cache:
      - cut first `cutoff_days` from IRENE
      - resample IRENE hourly means, stamp at :30
      - store ts_20min, ts_hourly, nearest_face_idx, bathy_at_points, tran_x/y
      - also save the mouth timeseries figure and peak times CSV (as before)
    """
    out_dir = CACHE_NC.parent
    if CACHE_NC.exists() and not force:
        print(f"[cache] Found existing {CACHE_NC.name} — keeping (set force=True to rebuild).")
        return

    print(f"[cache] Building {CACHE_NC.name} with {cutoff_days}-day cutoff and hourly means @ :30…")

    # Load transect CSV (ordered by est_name, mouth->river within each)
    tran = pd.read_csv(transect_csv).rename(columns={'X': 'x', 'Y': 'y'})
    tran = tran.sort_values(by='est_name', kind='stable').reset_index(drop=True)
    tran['group_id'] = pd.factorize(tran['est_name'], sort=True)[0]
    estuaries = tran['group_id'].unique()
    est_names = tran.groupby('group_id')['est_name'].first()
    tran_x = tran['x'].values
    tran_y = tran['y'].values

    # --- Open datasets
    regrid = xr.open_dataset(regrid_path)   # only for future Step 2 usage (grid tree prep not needed now)
    unstr  = xr.open_dataset(unstr_path)    # IRENE unstructured
    bathy_ds = xr.open_dataset(bathy_file)  # bed_face(mesh2d_nFaces)

    # --- Cut off first N days in IRENE
    time_coord = pick_first_coord(unstr, ['time'])
    t_full = pd.to_datetime(unstr[time_coord].values)
    cutoff_time = t_full[0] + pd.Timedelta(days=cutoff_days)
    unstr = unstr.sel({time_coord: slice(cutoff_time, None)})
    time_vals_20min = pd.to_datetime(unstr[time_coord].values)
    print(f"[cache] Trimmed start → {cutoff_time}")

    # --- Faces BallTree
    face_x_name = pick_first_existing(unstr, ['mesh2d_face_x'])
    face_y_name = pick_first_existing(unstr, ['mesh2d_face_y'])
    face_lon = unstr[face_x_name].values
    face_lat = unstr[face_y_name].values
    faces_tree = build_balltree_points(face_lon, face_lat)

    # --- IRENE var (time first)
    irene_name = pick_first_existing(unstr, ['mesh2d_s1','s1','surface_height','waterlevel','waterlevel_z','zeta'])
    irene = ensure_time_first(unstr[irene_name], time_coord)
    face_dim = find_face_dim(irene, time_coord)

    # --- Nearest face per transect point
    _, idx = faces_tree.query(deg2rad_xy(tran_x, tran_y), k=1)
    nearest_face_idx = idx[:, 0].astype(int)

    # --- Full 20-min timeseries at all points (time_20min, point)
    ts_20min = irene.isel({face_dim: xr.DataArray(nearest_face_idx, dims=("point",))})
    ts_20min = ts_20min.rename({ts_20min.dims[0]: "time_20min"})

    # --- RESAMPLE to hourly mean, then stamp at :30 and rename dim to "time"
    ts_hourly = ts_20min.resample({"time_20min": "60min"}).mean()
    time_vals_hourly = pd.to_datetime(ts_hourly["time_20min"].values) + pd.Timedelta(minutes=30)
    ts_hourly = ts_hourly.assign_coords(time=("time_20min", time_vals_hourly)).swap_dims({"time_20min": "time"}).drop_vars("time_20min")

    # --- Bathy at faces for each point
    bathy_name = pick_first_existing(bathy_ds, ['bed_face'])
    bathy_at_points = bathy_ds[bathy_name].isel(mesh2d_nFaces=xr.DataArray(nearest_face_idx, dims=("point",))).values

    # --- Write cache
    ds_out = xr.Dataset(
        data_vars=dict(
            ts_20min=(("time_20min","point"), ts_20min.values.astype(np.float32)),
            ts_hourly=(("time","point"), ts_hourly.values.astype(np.float32)),
            nearest_face_idx=(("point",), nearest_face_idx.astype(np.int32)),
            bathy_at_points=(("point",), bathy_at_points.astype(np.float32)),
            tran_x=(("point",), tran_x.astype(np.float64)),
            tran_y=(("point",), tran_y.astype(np.float64)),
        ),
        coords=dict(
            time_20min=("time_20min", time_vals_20min.astype("datetime64[ns]")),
            time=("time", time_vals_hourly.astype("datetime64[ns]")),
            point=("point", np.arange(len(tran_x), dtype=int)),
        ),
        attrs=dict(note="PRIMEA (IRENE) cache: 20-min + hourly@30min, faces + bathy, 10-day cutoff.")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(CACHE_NC)
    print(f"[cache] Wrote {CACHE_NC}")

    # ---- (nice to keep) mouth time-series fig + peak CSV, same as Step-1b
    n_est = len(estuaries)
    fig, axes = plt.subplots(n_est, 1, figsize=(10, 12), constrained_layout=True, sharex=True)
    if n_est == 1: axes = [axes]
    peaks = []
    tvh = pd.to_datetime(time_vals_hourly)

    for row_i, gid in enumerate(sorted(estuaries)):
        est_name = est_names.loc[gid]
        letter = int_to_letter(row_i)
        est_mask = (tran['group_id'].values == gid)
        est_points_idx = np.where(est_mask)[0]
        mouth_idx = est_points_idx[0]
        mouth_ts = ts_hourly.isel(point=mouth_idx).values
        k = int(np.nanargmax(mouth_ts)) if extreme == 'max' else int(np.nanargmin(mouth_ts))
        peak_time = pd.to_datetime(tvh[k])
        ax = axes[row_i]
        ax.plot(tvh, mouth_ts, lw=1.2, label=f"{est_name.capitalize()} mouth (hourly @ :30)")
        ax.axvline(peak_time, ls='--', lw=1.0)
        ax.set_ylabel("Surface height [m]")
        ax.set_title(f"{letter} — {est_name.capitalize()} • mouth series (IRENE hourly @ :30)")
        peaks.append(dict(group_id=int(gid), estuary=est_name,
                          mouth_face_index=int(nearest_face_idx[mouth_idx]),
                          mouth_lon=float(tran_x[mouth_idx]), mouth_lat=float(tran_y[mouth_idx]),
                          peak_type=extreme, peak_time_unstr_hourly_half=peak_time.isoformat(),
                          peak_value_hourly=float(mouth_ts[k])))
    axes[-1].set_xlabel("Time (UTC)")
    axes[0].legend(loc='upper right', fontsize=8)
    fig.savefig(out_dir / f"mouth_timeseries_{extreme.upper()}_HAVERSINE_CUTOFF_HOURLY.png",
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(peaks).to_csv(out_dir / f"mouth_peak_times_{extreme}_haversine_cutoff_hourly.csv",
                               index=False)
    print("[cache] Wrote mouth-timeseries figure and peak-times CSV.")

# ==================================================


for base in base_list:

    # Inputs
    regrid_path   = base / 'kent_regrid.nc'          # structured (UKC4)
    unstr_path    = base / 'kent_31_merged_map.nc'   # unstructured (IRENE)
    transect_csv  = Path('/media/af/PNC/modelling_DATA/kent_estuary_project/land_boundary/analysis/QGIS_shapefiles/points_along_estuary_1km_spacing.csv')
    bathy_file    = Path('/media/af/PNB_extra/Original_Data/interpolated_bathymetry_kent_31.nc')  # bed_face(mesh2d_nFaces)

    # High-res coastline outline shapefile (no basemap)
    outline_path  = Path('/media/af/PNC/modelling_DATA/kent_estuary_project/5.Final/QGIS/kent_area_poly_as_lines.shp')

    # Cache
    out_dir   = base / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    CACHE_NC  = out_dir / 'transect_timeseries_cache.nc'

    # ---- Build cache exactly as Step-1b ----
    build_cache_if_needed(
        CACHE_NC, regrid_path, unstr_path, transect_csv, bathy_file,
        extreme=EXTREME, cutoff_days=TRIM_FIRST_DAYS, force=FORCE_REBUILD_CACHE, dpi=DPI
    )

    # ---------- load cache ----------
    cache = xr.load_dataset(CACHE_NC)
    ts_hourly        = cache["ts_hourly"]                       # (time, point)  IRENE hourly @ :30
    ts_20min         = cache["ts_20min"]                        # (time_20min, point) IRENE 20min
    hourly_times     = pd.to_datetime(cache["time"].values)     # :30
    unstr_times_20m  = pd.to_datetime(cache["time_20min"].values)
    nearest_face_idx = cache["nearest_face_idx"].values.astype(int)
    bathy_at_points  = cache["bathy_at_points"].values
    tran_x           = cache["tran_x"].values
    tran_y           = cache["tran_y"].values

    # ---------- transect CSV (grouping + order) ----------
    tran = pd.read_csv(transect_csv).rename(columns={'X': 'x', 'Y': 'y'})
    tran = tran.sort_values(by='est_name', kind='stable').reset_index(drop=True)
    tran['group_id'] = pd.factorize(tran['est_name'], sort=True)[0]
    estuaries = np.unique(tran['group_id'])
    est_names = tran.groupby('group_id')['est_name'].first()

    # ---------- open raw datasets for sampling ----------
    unstr = xr.open_dataset(unstr_path)   # unstructured IRENE
    regr  = xr.open_dataset(regrid_path)  # structured UKC4

    # IRENE (unstructured) var + coords
    irene_unstr_name = pick_first_existing(unstr, ['mesh2d_s1', 's1', 'surface_height', 'waterlevel', 'waterlevel_z', 'zeta'])
    time_unstr_name  = pick_first_coord(unstr, ['time'])
    irene_unstr = ensure_time_first(unstr[irene_unstr_name], time_unstr_name)
    face_dim = find_face_dim(irene_unstr, time_unstr_name)
    unstr_times_all = unstr[time_unstr_name].values  # datetime64

    # Regrid lon/lat grid + UKC4 var/time
    lon_name = pick_first_coord(regr, ['nav_lon', 'lon', 'LONGITUDE'])
    lat_name = pick_first_coord(regr, ['nav_lat', 'lat', 'LATITUDE'])
    lon2d = regr[lon_name].values
    lat2d = regr[lat_name].values
    grid_tree, (ny, nx) = build_balltree_grid(lon2d, lat2d)

    ukc4_var_name   = pick_first_existing(regr, ['ukc4_surface_height', 'ukc4_surface_elevation', 'ukc4_sh'])
    ukc4_time_coord = pick_first_coord(regr, ['time_counter', 'time_instant'])  # hourly @ :30
    regrid_times    = regr[ukc4_time_coord].values

    # =====================================
    # Figure 1: along-transect profiles + stats
    # =====================================
    fig, axes = plt.subplots(len(estuaries), 1, figsize=FIGSIZE, constrained_layout=True, sharex=False)
    if len(estuaries) == 1:
        axes = [axes]

    # Legend above the figure (tight)
    fig.legend(
        handles=[
            Line2D([0],[0], color=CLR_IRENE,  lw=LINE_W, label="IRENE"),
            Line2D([0],[0], color=CLR_UKC4,   lw=LINE_W, label="UKC4"),
            Line2D([0],[0], color=CLR_BATHY,  lw=LINE_W, label="Bathymetry"),
        ],
        ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.015), frameon=True, fontsize=9
    )

    global_min = +np.inf
    global_max = -np.inf
    coverage_per_estuary = {}
    stats_rows = []

    for row_i, gid in enumerate(sorted(estuaries)):
        # Points (global indices) belonging to this estuary, CSV order (mouth->river)
        est_mask = (tran['group_id'].values == gid)
        pts_idx = np.where(est_mask)[0]
        est_name = est_names.loc[gid]
        letter = int_to_letter(row_i)

        # Peak time (mouth) on IRENE hourly series (from cache, already trimmed)
        mouth_idx = pts_idx[0]
        mouth_hourly = ts_hourly.isel(point=mouth_idx).values
        k = int(np.nanargmax(mouth_hourly)) if EXTREME == 'max' else int(np.nanargmin(mouth_hourly))
        peak_time_half = pd.to_datetime(hourly_times[k])

        # IRENE along-transect at peak hour (hourly mean = 3×20min average)
        i0, i1, i2 = three_20min_indices_for_hour(unstr_times_20m, peak_time_half)
        j0 = nearest_time_index(unstr_times_all, unstr_times_20m[i0])
        j1 = nearest_time_index(unstr_times_all, unstr_times_20m[i1])
        j2 = nearest_time_index(unstr_times_all, unstr_times_20m[i2])
        faces_est = nearest_face_idx[pts_idx]
        irene_3 = irene_unstr.isel({time_unstr_name: [j0, j1, j2],
                                    face_dim: xr.DataArray(faces_est, dims=("point",))}).values
        irene_line = np.nanmean(irene_3, axis=0)

        # UKC4 along-transect at nearest regrid time
        tidx = nearest_time_index(regrid_times, peak_time_half.to_datetime64())
        pts_lon = tran_x[pts_idx]
        pts_lat = tran_y[pts_idx]
        _, flat_idx = grid_tree.query(deg2rad_xy(pts_lon, pts_lat), k=1)
        flat_idx = flat_idx[:, 0].astype(int)
        ys = (flat_idx // nx).astype(int)
        xs = (flat_idx %  nx).astype(int)
        ukc4_slice = regr[ukc4_var_name].isel({ukc4_time_coord: tidx}).values  # (y,x)
        ukc4_line = ukc4_slice[ys, xs]

        # Coverage mask for outline figure
        ukc4_has = np.isfinite(ukc4_line)
        coverage_per_estuary[int(gid)] = (pts_idx, ukc4_has)

        # Bathymetry (UNSTRUCTURED)
        bathy_line = bathy_at_points[pts_idx]

        # Distance axis
        dist_km = cumulative_dist_km(pts_lon, pts_lat)

        # Stats per model (mouth & furthest upriver value)
        irene_valid_idx = np.where(np.isfinite(irene_line))[0]
        irene_up_idx = int(irene_valid_idx[-1]) if irene_valid_idx.size else np.nan
        irene_mouth = float(irene_line[0]) if np.isfinite(irene_line[0]) else np.nan
        irene_up = float(irene_line[irene_up_idx]) if irene_valid_idx.size else np.nan
        irene_up_dist = float(dist_km[irene_up_idx]) if irene_valid_idx.size else np.nan
        irene_delta = (irene_up - irene_mouth) if np.isfinite(irene_up) and np.isfinite(irene_mouth) else np.nan

        ukc4_valid_idx = np.where(np.isfinite(ukc4_line))[0]
        ukc4_up_idx = int(ukc4_valid_idx[-1]) if ukc4_valid_idx.size else np.nan
        ukc4_mouth = float(ukc4_line[0]) if np.isfinite(ukc4_line[0]) else np.nan
        ukc4_up = float(ukc4_line[ukc4_up_idx]) if ukc4_valid_idx.size else np.nan
        ukc4_up_dist = float(dist_km[ukc4_up_idx]) if ukc4_valid_idx.size else np.nan
        ukc4_delta = (ukc4_up - ukc4_mouth) if np.isfinite(ukc4_up) and np.isfinite(ukc4_mouth) else np.nan

        stats_rows.append({
            "letter": letter,
            "estuary": est_name,
            "timestep_utc": peak_time_half.strftime("%Y-%m-%d %H:%M"),
            "irene_mouth_m": irene_mouth,
            "irene_upriver_m": irene_up,
            "irene_upriver_point": irene_up_idx,
            "irene_upriver_dist_km": irene_up_dist,
            "irene_height_diff_m": irene_delta,
            "ukc4_mouth_m": ukc4_mouth,
            "ukc4_upriver_m": ukc4_up,
            "ukc4_upriver_point": ukc4_up_idx,
            "ukc4_upriver_dist_km": ukc4_up_dist,
            "ukc4_height_diff_m": ukc4_delta,
        })

        # Track limits
        cur = np.concatenate([np.ravel(irene_line), np.ravel(ukc4_line), np.ravel(bathy_line)])
        cur = cur[np.isfinite(cur)]
        if cur.size:
            global_min = min(global_min, cur.min())
            global_max = max(global_max, cur.max())

        # Plot
        ax = axes[row_i]
        ax.plot(dist_km, irene_line, label='IRENE', linewidth=LINE_W, color=CLR_IRENE)
        ax.plot(dist_km, ukc4_line,  label='UKC4',  linewidth=LINE_W, color=CLR_UKC4)
        ax.plot(dist_km, bathy_line, label='Bathymetry', linewidth=LINE_W, color=CLR_BATHY)

        # Title
        ax.set_title(f"{letter} ({est_name.capitalize()}; timestep: {peak_time_half:%Y-%m-%d %H:%M} UTC)")
        ax.set_ylabel("Height [m]")
        if row_i == len(estuaries) - 1:
            ax.set_xlabel("Distance along estuary [km]")

    # Harmonize y-lims
    if np.isfinite(global_min) and np.isfinite(global_max):
        pad = 0.05 * max(1e-9, (global_max - global_min))
        for ax in axes:
            ax.set_ylim(global_min - pad, global_max + pad)

    out_png = out_dir / f"transects_phase_aligned_{EXTREME.upper()}.png"
    fig = axes[0].get_figure()
    fig.savefig(out_png, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved transect figure: {out_png}")

    # ---------- stats CSV ----------
    stats_df = pd.DataFrame(stats_rows, columns=[
        "letter","estuary","timestep_utc",
        "irene_mouth_m","irene_upriver_m","irene_upriver_point","irene_upriver_dist_km","irene_height_diff_m",
        "ukc4_mouth_m","ukc4_upriver_m","ukc4_upriver_point","ukc4_upriver_dist_km","ukc4_height_diff_m",
    ])
    for col in ["irene_upriver_point", "ukc4_upriver_point"]:
        stats_df[col] = stats_df[col].astype("Int64")

    stats_csv = out_dir / f"transect_stats_{EXTREME}.csv"
    stats_df.to_csv(stats_csv, index=False, float_format="%.3f")
    print(f"Saved stats CSV: {stats_csv}")

    # =====================================
    # Exact UKC4 outline (cell-edge, no-NaNs)
    # =====================================
    print("Computing exact UKC4 outline from valid cells (any-time finite) …")
    ukc4_mask = np.isfinite(regr[ukc4_var_name]).any(dim=ukc4_time_coord).values  # (ny, nx)
    ukc4_geom = outline_curvilinear_from_mask(lon2d, lat2d, ukc4_mask)
    if ukc4_geom is None:
        raise RuntimeError("UKC4 mask produced no valid cells; cannot build outline.")

    gdf_ukc4 = gpd.GeoDataFrame({"model": ["UKC4"], "geometry": [ukc4_geom]}, crs="EPSG:4326")
    ukc4_shp = out_dir / "ukc4_outline_exact.shp"
    gdf_ukc4.to_file(ukc4_shp)
    print(f"Saved exact UKC4 outline: {ukc4_shp}")

    # =====================================
    # Figure 2: outline with coloured transect points (no basemap)
    # =====================================
    print("Building outline figure (coverage-coloured points + exact UKC4 & IRENE boundaries)…")

    # Coastline outline (black)
    outline = gpd.read_file(outline_path)

    fig2, ax2 = plt.subplots(figsize=(8, 10))
    outline.plot(ax=ax2, facecolor='none', edgecolor='black', linewidth=LINE_W, zorder=1)

    # UKC4 boundary (orange)
    gdf_ukc4.plot(ax=ax2, facecolor='none', edgecolor=CLR_UKC4, linewidth=LINE_W, zorder=2)

    # ===== IRENE boundary (blue) from unstructured grid =====
    start_path = Path(opsys("PNC"))
    main_path = start_path / "modelling_DATA/kent_estuary_project/5.Final/1.friction/2.0.1_wind_testing_4_months_5_second_timestep.dsproj_data"
    bathy_path = next((main_path / "FlowFM").glob("*.nc"), None)

    def _np(a):
        return a.values if hasattr(a, "values") else np.asarray(a)

    def _grid(ds):
        ugc = ds.ugrid
        if hasattr(ugc, "grid"):
            return ugc.grid
        if hasattr(ugc, "grids") and len(ugc.grids) > 0:
            return ugc.grids[0]
        raise RuntimeError("No UGRID grid found (.grid/.grids).")

    def _face_nodes_and_fill(ug):
        fna = ug.face_node_connectivity
        arr = _np(fna)
        fill = getattr(fna, "fill_value", None)
        if fill is None:
            fill = getattr(fna, "_FillValue", None)
        if fill is None and hasattr(fna, "encoding"):
            fill = fna.encoding.get("_FillValue", None)
        return arr, (-1 if fill is None else int(fill))

    def _faces_list(face_nodes, fill_value):
        return [row[row != fill_value] for row in face_nodes]

    def _boundary_segments(node_xy, faces_list, include_face):
        edge_counts = {}
        directed = []
        for f_idx, nodes in enumerate(faces_list):
            if not include_face[f_idx] or len(nodes) < 3:
                continue
            m = len(nodes)
            for k in range(m):
                u = int(nodes[k]); v = int(nodes[(k + 1) % m])
                key = (u, v) if u < v else (v, u)
                edge_counts[key] = edge_counts.get(key, 0) + 1
                directed.append((u, v))
        segs = []
        for (u, v) in directed:
            key = (u, v) if u < v else (v, u)
            if edge_counts.get(key, 0) == 1:
                x1, y1 = node_xy[u]
                x2, y2 = node_xy[v]
                segs.append((float(x1), float(y1), float(x2), float(y2)))
        return segs

    uds = xu.open_dataset(bathy_path)
    ug = _grid(uds)
    node_xy = np.column_stack([_np(ug.node_x), _np(ug.node_y)])
    face_nodes, fill_value = _face_nodes_and_fill(ug)
    faces_list = _faces_list(face_nodes, fill_value)

    bathy = uds["mesh2d_node_z"]
    if bathy.ndim > 1:
        sel = {d: 0 for d in bathy.dims if d != ug.node_dimension}
        bathy = bathy.isel(**sel)
    zn = _np(bathy)
    include_valid = np.array([np.isfinite(zn[f]).all() for f in faces_list], dtype=bool)

    segments = _boundary_segments(node_xy, faces_list, include_valid)
    lines = [LineString([(x1, y1), (x2, y2)]) for (x1, y1, x2, y2) in segments]
    merged = linemerge(unary_union(lines))
    polys = list(polygonize(merged))
    if not polys:
        xs = [s[0] for s in segments] + [s[2] for s in segments]
        ys = [s[1] for s in segments] + [s[3] for s in segments]
        eps = 1e-4 * max((max(xs) - min(xs)), (max(ys) - min(ys)))
        filled = unary_union(lines).buffer(eps)
        if isinstance(filled, Polygon):
            polys = [filled]
        elif isinstance(filled, MultiPolygon):
            polys = list(filled.geoms)
    outer = max(polys, key=lambda p: p.area) if polys else None

    if outer is not None:
        outer = outer.buffer(0)
        xx, yy = outer.exterior.xy
        ax2.plot(xx, yy, color=CLR_IRENE, linewidth=LINE_W, zorder=10)

    # ===== Transect points & lines =====
    legend_elems = [
        Line2D([0], [0], color='black', lw=LINE_W, label='Coastline'),
        Line2D([0], [0], color=CLR_UKC4,  lw=LINE_W, label='UKC4 boundary'),
        Line2D([0], [0], color=CLR_IRENE, lw=LINE_W, label='IRENE boundary'),
        Line2D([0], [0], marker='o', linestyle='None', markerfacecolor=CLR_UKC4, markeredgecolor='k',
               label='UKC4/IRENE transect', markersize=6),
        Line2D([0], [0], marker='o', linestyle='None', markerfacecolor=CLR_IRENE, markeredgecolor='k',
               label='IRENE transect', markersize=6),
    ]

    dx_top, dy_top = 0.006, 0.006           # default top-right
    dx_bottom, dy_bottom = -0.006, -0.006   # for C, D bottom-left

    for row_i, gid in enumerate(sorted(estuaries)):
        pts_idx, ukc4_has = coverage_per_estuary[int(gid)]
        est_pts_x = tran_x[pts_idx]
        est_pts_y = tran_y[pts_idx]
        colors = np.where(ukc4_has, CLR_UKC4, CLR_IRENE)

        ax2.scatter(est_pts_x, est_pts_y, c=colors, s=9, zorder=3,
                    edgecolors='k', linewidths=0.4)
        ax2.scatter(est_pts_x[0], est_pts_y[0], marker='D', s=20,
                    facecolors='none', edgecolors='k', zorder=4, linewidths=0.8)
        ax2.plot(est_pts_x, est_pts_y, color='k', alpha=0.25, linewidth=LINE_W, zorder=2)

        letter = int_to_letter(row_i)
        if letter in ['C', 'D']:
            dx, dy = dx_bottom, dy_bottom; ha, va = 'right', 'top'
        else:
            dx, dy = dx_top, dy_top; ha, va = 'left', 'bottom'
        ax2.text(est_pts_x[0] + dx, est_pts_y[0] + dy, letter, fontsize=9,
                 fontweight='bold', ha=ha, va=va, color='k', zorder=5)

        est_name = est_names.loc[gid].capitalize()
        legend_elems.append(
            Line2D([0], [0],
                   marker=r'$\mathdefault{' + letter + '}$',
                   markersize=10, linestyle='None', color='k',
                   label=f"{est_name}")
        )

    ax2.set_xlim(-3.6, -2.575)
    ax2.set_ylim(53.175, 54.5)
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(True, alpha=0.25)

    ax2.legend(handles=legend_elems, loc='center left',
               bbox_to_anchor=(0.02, 0.45), frameon=True)

    out_png2 = out_dir / f"transects_points_outline_{EXTREME.upper()}.png"
    fig2.savefig(out_png2, dpi=DPI, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved outline figure: {out_png2}")
