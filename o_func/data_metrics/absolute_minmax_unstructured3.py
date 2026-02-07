#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Storm-surge snapshot for unstructured UGRID (with caching):
- COMPUTE (once): find the timestamp with the largest spatial surface-height metric
  and save "storm_surge_snapshot.nc" (mesh + sh_at_peak + time_peak + metadata).
- RE-USE: if that file exists and OVERWRITE=False, skip recompute.
- PLOT: render the snapshot; optional bathymetry-based dry overlay.

Toggle bathy overlay (PLOT_BED_OVERLAY), metric (mean / p90), and optional coastal box.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmocean as cmo
from scipy.interpolate import griddata

import matplotlib
matplotlib.use("Agg")
from matplotlib.collections import PolyCollection

# =================== USER SETTINGS ===================

BED_XYZ_PATH = Path('/Volumes/PNC') / 'modelling_DATA/kent_estuary_project/5.Final/1.friction/2.0.1_wind_testing_4_months_5_second_timestep.dsproj_data' / 'bed_level_deepened_channel(testing).xyz'

SURGE_METRIC = "mean"      # "mean" or "p90"
COAST_BOX    = None        # (xmin, xmax, ymin, ymax) or None for full domain

X_LIMITS = (-3.6, -2.75)
Y_LIMITS = None

SSH_CLIM = (0, 10)
SSH_CMAP = cmo.cm.dense

PLOT_BED_OVERLAY = True
DRY_THRESH_M     = 0.02
DRY_COLOR        = "#8B5A2B"

ENGINE = "netcdf4"
SURFACE_HEIGHT_NAMES = ("mesh2d_s1", "waterlevel", "water_level", "zeta")

# =====================================================

# --------- UGRID helpers ----------
def _list_mesh_topologies(ds):
    tops = []
    for v in ds.variables:
        if ds[v].attrs.get("cf_role", "").lower() == "mesh_topology":
            tops.append(v)
    if not tops:
        raise RuntimeError("No UGRID mesh_topology variables found.")
    return tops

def _repair_connectivity_da(da, n_nodes=None):
    a = np.asarray(da.data).copy()
    if np.issubdtype(a.dtype, np.floating):
        a = np.where(np.isnan(a), -1, np.rint(a))
    a = a.astype("int64", copy=False)
    a[a < 0] = -1
    a = a.astype("int32", copy=False)
    nonfill = (a >= 0)
    if nonfill.any() and int(a[nonfill].min()) == 1:
        a[nonfill] = a[nonfill] - 1
    if n_nodes is not None and n_nodes > 0:
        over = nonfill & (a >= n_nodes)
        if over.any():
            a[over] = n_nodes - 1
    out = xr.DataArray(a, dims=da.dims, coords=da.coords, name=da.name,
                       attrs={**da.attrs, "start_index": 0})
    out.encoding = {**da.encoding, "_FillValue": -1}
    return out

def _sanitize_topology(ds: xr.Dataset) -> xr.Dataset:
    tops = _list_mesh_topologies(ds)
    n_nodes = None
    for cand in ("mesh2d_node_x", "mesh2d_node_y"):
        if cand in ds:
            n_nodes = int(ds[cand].shape[0]); break
    for topo in tops:
        tvar = ds[topo]
        face_name = tvar.attrs.get("face_node_connectivity") or ("mesh2d_face_nodes" if "mesh2d_face_nodes" in ds else None)
        edge_name = tvar.attrs.get("edge_node_connectivity") or ("mesh2d_edge_nodes" if "mesh2d_edge_nodes" in ds else None)
        if face_name:
            ds[face_name] = _repair_connectivity_da(ds[face_name], n_nodes=n_nodes)
        if edge_name and edge_name in ds:
            ds[edge_name] = _repair_connectivity_da(ds[edge_name], n_nodes=n_nodes)
    for v in ds.variables:
        if "connectivity" in v or v in ("mesh2d_face_nodes", "mesh2d_edge_nodes"):
            ds[v].attrs["start_index"] = 0
            ds[v].encoding["_FillValue"] = -1
    return ds

def _compute_face_centroids(face_nodes, node_x, node_y):
    nfaces = face_nodes.shape[0]
    cx = np.full(nfaces, np.nan, float)
    cy = np.full(nfaces, np.nan, float)
    for i in range(nfaces):
        idx = face_nodes[i]; idx = idx[idx >= 0]
        if idx.size >= 3:
            xs = node_x[idx]; ys = node_y[idx]
            if np.all(np.isfinite(xs)) and np.all(np.isfinite(ys)):
                cx[i] = xs.mean(); cy[i] = ys.mean()
    return cx, cy

def _tighten_axes(ax, node_x, node_y, xlim=None, ylim=None, pad=0.01):
    if xlim is None: xmin, xmax = np.nanmin(node_x), np.nanmax(node_x)
    else:            xmin = xlim[0] if xlim[0] is not None else np.nanmin(node_x); xmax = xlim[1] if xlim[1] is not None else np.nanmax(node_x)
    if ylim is None: ymin, ymax = np.nanmin(node_y), np.nanmax(node_y)
    else:            ymin = ylim[0] if ylim[0] is not None else np.nanmin(node_y); ymax = ylim[1] if ylim[1] is not None else np.nanmax(node_y)
    dx = xmax - xmin; dy = ymax - ymin
    ax.set_xlim(xmin - pad*dx, xmax + pad*dx)
    ax.set_ylim(ymin - pad*dy, ymax + pad*dy)

def _poly_from_faces(face_nodes, node_x, node_y, mask=None, stride=1):
    polys, idxs = [], []
    nfaces = face_nodes.shape[0]
    mask = np.asarray(mask, dtype=bool) if mask is not None else None
    for i in range(0, nfaces, max(1, stride)):
        if mask is not None and not mask[i]: continue
        fi = face_nodes[i]; fi = fi[fi >= 0]
        if fi.size < 3: continue
        xs = node_x[fi]; ys = node_y[fi]
        if np.any(~np.isfinite(xs)) or np.any(~np.isfinite(ys)): continue
        polys.append(np.column_stack([xs, ys])); idxs.append(i)
    return polys, np.asarray(idxs, dtype=int)

# --------- Bathy interpolation ----------
def _interpolate_bathy_to_faces(xyz_path: Path, node_x, node_y, face_nodes):
    arr = np.loadtxt(xyz_path, dtype=float)
    if arr.shape[1] < 3:
        raise ValueError("XYZ must have 3 columns: lon lat z")
    lon, lat, zb = arr[:, 0], arr[:, 1], arr[:, 2]
    cx, cy = _compute_face_centroids(face_nodes, node_x, node_y)
    bed_face = griddata(points=(lon, lat), values=zb, xi=np.column_stack([cx, cy]),
                        method="linear")
    return bed_face

# --------- Storm-surge compute (with caching) ----------
def compute_storm_surge_snapshot(
    ds_path: str,
    out_dir: str,
    engine: str = ENGINE,
    height_names=SURFACE_HEIGHT_NAMES,
    coast_box=None,
    metric: str = "mean",
    overwrite: bool = False,
) -> Path:
    """
    Find the timestamp with the largest spatial surface-height metric and
    save mesh + sh_at_peak + time_peak to storm_surge_snapshot.nc.
    If the file exists and overwrite=False, skip recompute.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "storm_surge_snapshot.nc"

    if out_path.exists() and not overwrite:
        print(f"[SKIP] Snapshot exists: {out_path} (set overwrite=True to recompute)")
        return out_path

    print(f"[STORM] Open: {ds_path}")
    ds = xr.open_dataset(ds_path, engine=engine)
    ds = _sanitize_topology(ds)

    sh = next((ds[n] for n in height_names if n in ds), None)
    if sh is None:
        raise KeyError(f"None of {height_names} found in dataset.")
    if "time" not in sh.dims:
        raise ValueError(f"{sh.name} has no 'time' dimension; need full time series.")
    sh = sh.astype("float32")

    face_nodes_name = ds[_list_mesh_topologies(ds)[0]].attrs.get("face_node_connectivity") or "mesh2d_face_nodes"
    face_nodes = ds[face_nodes_name].values.astype(int)
    node_x = ds["mesh2d_node_x"].values.astype(float)
    node_y = ds["mesh2d_node_y"].values.astype(float)
    cx, cy = _compute_face_centroids(face_nodes, node_x, node_y)

    # Coast box -> which faces contribute to time-series metric
    if coast_box is not None:
        xmin, xmax, ymin, ymax = coast_box
        in_box = (cx >= xmin) & (cx <= xmax) & (cy >= ymin) & (cy <= ymax)
        if not np.any(in_box):
            print("[STORM] Warning: COAST_BOX selects zero faces; using full domain.")
            in_box = np.ones_like(cx, dtype=bool)
    else:
        in_box = np.ones_like(cx, dtype=bool)

    # Metric time series
    sh_sub = sh.sel(mesh2d_nFaces=in_box) if "mesh2d_nFaces" in sh.dims else sh
    if metric.lower() == "p90":
        series = sh_sub.quantile(0.90, dim=[d for d in sh_sub.dims if d != "time"], skipna=True)
    else:
        series = sh_sub.mean(dim=[d for d in sh_sub.dims if d != "time"], skipna=True)

    t_idx = int(series.argmax("time").values)
    t_peak = pd.to_datetime(sh["time"].values[t_idx])
    print(f"[STORM] Peak time: {t_peak.isoformat()} (metric={metric}, coast_box={'set' if coast_box else 'full'})")

    sh_at_peak = sh.isel(time=t_idx).squeeze()
    for d in list(sh_at_peak.dims):
        if d != "mesh2d_nFaces":
            sh_at_peak = sh_at_peak.mean(d)

    # Save compact snapshot + provenance
    snap = xr.Dataset(
        data_vars=dict(
            sh_at_peak=sh_at_peak.astype("float32"),
            mesh2d_node_x=(("mesh2d_nNodes",), node_x.astype("float64")),
            mesh2d_node_y=(("mesh2d_nNodes",), node_y.astype("float64")),
            mesh2d_face_nodes=(("mesh2d_nFaces","max_nodes_per_face"),
                               face_nodes.astype("int32")),
        ),
        coords=dict(time_peak=np.array([np.datetime64(t_peak)], dtype="datetime64[ns]")),
        attrs=dict(
            source_ds=str(ds_path),
            surge_metric=metric,
            coast_box=str(coast_box) if coast_box is not None else "full_domain",
        ),
    )
    snap["mesh2d"] = xr.DataArray(0, attrs=dict(
        cf_role="mesh_topology",
        topology_dimension=2,
        node_coordinates="mesh2d_node_x mesh2d_node_y",
        face_node_connectivity="mesh2d_face_nodes",
    ))
    snap["mesh2d_face_nodes"].attrs["start_index"] = 0
    snap["mesh2d_face_nodes"].encoding["_FillValue"] = -1

    enc = {k: dict(zlib=True, complevel=4) for k in snap.data_vars}
    snap.to_netcdf(out_path, engine="netcdf4", encoding=enc)
    print(f"[STORM] Wrote snapshot: {out_path}")
    return out_path

# --------- Plotting ----------
def _plot_snapshot_with_optional_bed(
    sh_1d, node_x, node_y, face_nodes, out_png, title,
    clim=SSH_CLIM, cmap=SSH_CMAP, xlim=None, ylim=None,
    plot_bed_overlay=True, dry_thresh=DRY_THRESH_M, dry_color=DRY_COLOR,
):
    fig, ax = plt.subplots(figsize=(6.0, 8.0))
    ax.set_facecolor("lightgrey")

    polys, idx = _poly_from_faces(face_nodes, node_x, node_y, mask=None, stride=1)
    vals = np.asarray(sh_1d).reshape(-1)
    coll = PolyCollection(polys, array=vals[idx], cmap=cmap)
    coll.set_rasterized(True)
    if clim is not None:
        coll.set_clim(*clim)
    ax.add_collection(coll)
    cbar = fig.colorbar(coll, ax=ax)
    cbar.set_label("Surface Height [m]")

    # optional dry overlay
    if plot_bed_overlay:
        print("[PLOT] Computing bathymetry overlay…")
        bed = _interpolate_bathy_to_faces(BED_XYZ_PATH, node_x, node_y, face_nodes)
        depth = vals - bed
        dry = np.isfinite(depth) & (depth <= dry_thresh)
        if np.any(dry):
            dry_polys, _ = _poly_from_faces(face_nodes, node_x, node_y, mask=dry, stride=1)
            overlay = PolyCollection(dry_polys, facecolor=dry_color, edgecolor="none", alpha=0.95)
            overlay.set_rasterized(True)
            ax.add_collection(overlay)
            print(f"[PLOT] Dry faces: {int(np.sum(dry))}")

    _tighten_axes(ax, node_x, node_y, xlim=xlim, ylim=ylim)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=450)

def plot_storm_snapshot(snapshot_nc: Path, out_dir: Path,
                        x_limits=None, y_limits=None,
                        plot_bed_overlay=PLOT_BED_OVERLAY):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ds = xr.open_dataset(snapshot_nc, engine="netcdf4")

    node_x    = ds["mesh2d_node_x"].values
    node_y    = ds["mesh2d_node_y"].values
    face_nodes= ds["mesh2d_face_nodes"].values.astype(int)
    sh_1d     = ds["sh_at_peak"].values
    t_peak    = pd.to_datetime(ds["time_peak"].values[0])

    # pull title metadata if present
    metric   = ds.attrs.get("surge_metric", "mean")
    cbox_txt = ds.attrs.get("coast_box", "full_domain")
    title = f"Storm-surge snapshot — {t_peak:%Y-%m-%d %H:%M UTC}  (metric={metric}, region={cbox_txt})"

    _plot_snapshot_with_optional_bed(
        sh_1d, node_x, node_y, face_nodes,
        out_png = out_dir / "storm_surge_surface_height.png",
        title   = title,
        clim    = SSH_CLIM, cmap=SSH_CMAP,
        xlim=x_limits, ylim=y_limits,
        plot_bed_overlay=plot_bed_overlay,
    )
    print("[PLOT] Wrote storm_surge_surface_height.png")

# ----------------- Runner -----------------
if __name__ == "__main__":
    OVERWRITE = False  # <-- set True to force recompute of the snapshot

    runs = [
        ("/Volumes/PNB_extra/runSCW_ao_yawind_orig8RealRiver_m0.035_Forcing_85_Discouv-58757607/kent_31_merged_map.nc",
         "/Volumes/PNB_extra/runSCW_ao_yawind_orig8RealRiver_m0.035_Forcing_85_Discouv-58757607/exports_singlepoint"),
        ("/Volumes/PNB_extra/runSCW_ao_yawind_AllRivNoDuddonClimatology_m0.035_Forcing_85_Discouv-58757598/kent_31_merged_map.nc",
         "/Volumes/PNB_extra/runSCW_ao_yawind_AllRivNoDuddonClimatology_m0.035_Forcing_85_Discouv-58757598/exports_singlepoint"),
        ("/Volumes/PNB_extra/scw_10layer_realriv/kent_31_merged_map.nc",
         "/Volumes/PNB_extra/scw_10layer_realriv/exports_singlepoint"),
        ("/Volumes/PNB_extra/scw_10_layer_climatology/kent_31_merged_map.nc",
         "/Volumes/PNB_extra/scw_10_layer_climatology/exports_singlepoint"),
    ]

    for DS_PATH, OUT_DIR in runs:
        print(f"\n=== Storm surge detect & plot ===\nDataset: {DS_PATH}\nOut:     {OUT_DIR}\n")
        snap_path = compute_storm_surge_snapshot(
            ds_path=DS_PATH,
            out_dir=OUT_DIR,
            engine=ENGINE,
            height_names=SURFACE_HEIGHT_NAMES,
            coast_box=COAST_BOX,
            metric=SURGE_METRIC,
            overwrite=OVERWRITE,
        )
        plot_storm_snapshot(
            snapshot_nc=snap_path,
            out_dir=Path(OUT_DIR),
            x_limits=X_LIMITS,
            y_limits=Y_LIMITS,
            plot_bed_overlay=PLOT_BED_OVERLAY,
        )

