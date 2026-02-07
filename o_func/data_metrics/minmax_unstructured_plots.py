#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UGRID min/max (surface-only) for Spyder:
- COMPUTE once -> saves a small results NetCDF beside your plots
- PLOT many times -> tweak clims/colormaps/limits without recomputing
"""

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import cmocean as cmo
import matplotlib.pyplot as plt

# cmap_topo = cmo.cm.deep_r
# mpl.colormaps
cmap_topo = cmo.cm.dense
cmap_sal = cmo.cm.haline_r

import matplotlib
matplotlib.use("Agg")  # comment out if you want interactive plots
from matplotlib.collections import PolyCollection
from matplotlib.cm import get_cmap

# Optional libs
try:
    import xugrid as xu
    _HAS_XUGRID = True
except Exception:
    _HAS_XUGRID = False

try:
    import cmocean
    _HAS_CMO = True
except Exception:
    _HAS_CMO = False


# ========= User defaults (edit) =========
# Run the 10realriv
# DS_PATH       = "/Volumes/PNB_extra/scw_10layer_realriv/kent_31_merged_map.nc"
# OUT_DIR       = "/Volumes/PNB_extra/scw_10layer_realriv/exports_singlepoint"
# Run the 10clim
# DS_PATH       = "/Volumes/PNB_extra/scw_10_layer_climatology/kent_31_merged_map.nc"
# OUT_DIR       = "/Volumes/PNB_extra/scw_10_layer_climatology/exports_singlepoint"
# Run the 1realriv
# DS_PATH       = "/Volumes/PNB_extra/runSCW_ao_yawind_orig8RealRiver_m0.035_Forcing_85_Discouv-58757607/kent_31_merged_map.nc"
# OUT_DIR       = "/Volumes/PNB_extra/runSCW_ao_yawind_orig8RealRiver_m0.035_Forcing_85_Discouv-58757607/exports_singlepoint"
# Run the 1clim
# DS_PATH_main       = "/Volumes/PNB_extra/runSCW_ao_yawind_AllRivNoDuddonClimatology_m0.035_Forcing_85_Discouv-58757598/kent_31_merged_map.nc"
# OUT_DIR_main      = "/Volumes/PNB_extra/runSCW_ao_yawind_AllRivNoDuddonClimatology_m0.035_Forcing_85_Discouv-58757598/exports_singlepoint"


RESULTS_NAME  = "results_minmax_surface.nc"   # saved inside OUT_DIR
ENGINE        = "netcdf4"
DONOR_PATH    = None        # optional donor for connectivity

# Spin-up & rolling windows
HEIGHT_SPINUP_DAYS   = 7
SALINITY_SPINUP_DAYS = 60
HEIGHT_WINDOW_H      = 3
SALINITY_WINDOW_H    = 24

# Plot options
HEIGHT_CLIM      = (-2, 6)     # set None to use recommended symmetric auto (printed)
SALINITY_CLIM    = (0, 35)     # fixed; yellow=fresh -> blue=salty

# Axis limits (lon/lat). Use None to auto from mesh. You can pass one-sided limits too.
# Example to push the ocean boundary to the left edge: X_LIMITS = (-3.6, None)
X_LIMITS = (-3.6, -2.75)  # e.g., (-3.6, -2.75)
Y_LIMITS = None  # e.g., (53.2, 54.5)

# Fallback PolyCollection decimation (for huge meshes)
POLY_STRIDE = 1   # draw every Nth face; try 2/4 if memory is tight

# Variables to look for
SURFACE_HEIGHT_NAMES = ("mesh2d_s1", "waterlevel", "water_level", "zeta")
SALINITY_3D_NAMES    = ("mesh2d_sa", "salinity_3d", "sa")
SALINITY_2D_NAMES    = ("mesh2d_sa1", "salinity", "salinity_2d", "sa1")


# ========= UGRID connectivity helpers =========
def _list_mesh_topologies(ds):
    tops = []
    for v in ds.variables:
        if ds[v].attrs.get("cf_role", "").lower() == "mesh_topology":
            tops.append(v)
    if hasattr(ds, "ugrid_roles"):
        try:
            for t in ds.ugrid_roles.topology:
                if t not in tops:
                    tops.append(t)
        except Exception:
            pass
    if not tops:
        raise RuntimeError("No UGRID mesh_topology variables found.")
    return tops

def _summarize_connectivity(label, da):
    arr = np.asarray(da.data)
    negs = arr[arr < 0]
    nonfill = arr[arr >= 0]
    print(f"[INFO] {label}: shape={arr.shape}, dtype={arr.dtype}, "
          f"start_index={da.attrs.get('start_index')}, "
          f"fill={da.encoding.get('_FillValue', None)}")
    if negs.size:
        print(f"       negatives (sample): {np.unique(negs)[:10]}")
    else:
        print("       negatives: none")
    if nonfill.size:
        print(f"       min/max non-fill: {int(nonfill.min())} / {int(nonfill.max())}")

def _repair_connectivity_da(da, n_nodes=None, label="connectivity"):
    a = np.asarray(da.data).copy()
    _summarize_connectivity(f"{label} (pre)", da)

    if np.issubdtype(a.dtype, np.floating):
        nanmask = np.isnan(a)
        if nanmask.any():
            a[nanmask] = -1
        a = np.rint(a)

    if np.issubdtype(a.dtype, np.integer):
        try:
            imin = np.iinfo(a.dtype).min
            a[a == imin] = -1
        except Exception:
            pass

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

    if (a < 0).any() and not np.all(a[a < 0] == -1):
        bad = np.unique(a[(a < 0) & (a != -1)])[:10]
        raise ValueError(f"{label}: unexpected negatives after repair: {bad}")

    out = xr.DataArray(
        a, dims=da.dims, coords=da.coords, name=da.name,
        attrs={**da.attrs, "start_index": 0},
    )
    out.encoding = {**da.encoding, "_FillValue": -1}
    _summarize_connectivity(f"{label} (post)", out)
    return out

def _adopt_from_donor(base, donor, face_name, edge_name, topo_name):
    if face_name and face_name in donor and face_name in base:
        if base[face_name].shape != donor[face_name].shape:
            raise ValueError(f"[{topo_name}] face shape mismatch: {base[face_name].shape} vs {donor[face_name].shape}")
        base[face_name] = donor[face_name].copy(deep=True)
        base[face_name].attrs["start_index"] = donor[face_name].attrs.get("start_index", 0)
        base[face_name].encoding["_FillValue"] = donor[face_name].encoding.get("_FillValue", -1)
        _summarize_connectivity(f"{topo_name} face (donor copied)", base[face_name])
    if edge_name and edge_name in donor and edge_name in base and base[edge_name].shape == donor[edge_name].shape:
        base[edge_name] = donor[edge_name].copy(deep=True)
        base[edge_name].attrs["start_index"] = donor[edge_name].attrs.get("start_index", 0)
        base[edge_name].encoding["_FillValue"] = donor[edge_name].encoding.get("_FillValue", -1)
        _summarize_connectivity(f"{topo_name} edge (donor copied)", base[edge_name])
    return base

def _sanitize_all_topologies(base: xr.Dataset, donor_path: str | None = None) -> xr.Dataset:
    tops = _list_mesh_topologies(base)
    print(f"[INFO] Found mesh topologies: {tops}")
    donor = xr.open_dataset(donor_path) if donor_path else None

    # node count for bounds
    n_nodes = None
    for cand in ("mesh2d_node_x", "mesh2d_node_y"):
        if cand in base:
            n_nodes = int(base[cand].shape[0])
            break

    for topo in tops:
        tvar = base[topo]
        face_name = tvar.attrs.get("face_node_connectivity") or ("mesh2d_face_nodes" if "mesh2d_face_nodes" in base else None)
        edge_name = tvar.attrs.get("edge_node_connectivity") or ("mesh2d_edge_nodes" if "mesh2d_edge_nodes" in base else None)

        print(f"[INFO] Topology '{topo}': face='{face_name}', edge='{edge_name}'")

        if donor is not None:
            base = _adopt_from_donor(base, donor, face_name, edge_name, topo)

        if face_name:
            base[face_name] = _repair_connectivity_da(base[face_name], n_nodes=n_nodes, label=f"{topo} face")
        if edge_name and edge_name in base:
            base[edge_name] = _repair_connectivity_da(base[edge_name], n_nodes=n_nodes, label=f"{topo} edge")

    # normalize attrs
    for v in base.variables:
        if "connectivity" in v or v in ("mesh2d_face_nodes", "mesh2d_edge_nodes"):
            base[v].attrs["start_index"] = 0
            base[v].encoding["_FillValue"] = -1

    print("[INFO] All topologies repaired and normalised.")
    return base


# ========= time / rolling / surface helpers =========
def _ensure_time(da):
    if "time" not in da.dims:
        raise ValueError(f"{da.name}: no 'time' dimension")
    if not np.issubdtype(da["time"].dtype, np.datetime64):
        da = da.assign_coords(time=pd.to_datetime(da["time"].values))
    return da

def _hours_to_samples(da, hours):
    t = pd.to_datetime(da["time"].values)
    if t.size < 2:
        return 1
    diffs = np.diff(t.astype("datetime64[ns]").astype(np.int64))
    dt_sec = np.median(diffs) / 1e9
    if dt_sec <= 0:
        dt_sec = 1200.0  # fallback 20 min
    samples = int(max(1, round((hours * 3600.0) / float(dt_sec))))
    return samples

def _rolling_extreme_mean(da, hours, extreme):
    da = _ensure_time(da)
    win = _hours_to_samples(da, hours)
    print(f"[INFO] Rolling window for {da.name}: {hours}h -> {win} samples")
    rol = da.rolling(time=win, center=True, min_periods=1)
    out = rol.max() if extreme == "max" else rol.min()
    return out.mean("time")

def _slice_spinup(da, days):
    da = _ensure_time(da)
    t0 = pd.to_datetime(da.time.values[0])
    return da.sel(time=slice(t0 + pd.Timedelta(days=days), None))

def _surface_layer_index(ds):
    if "mesh2d_layer_sigma" not in ds:
        return 0
    s = np.asarray(ds["mesh2d_layer_sigma"].values)
    return int(np.argmax(s))  # surface-most

def _force_surface(da, ds):
    return da.isel(mesh2d_nLayers=_surface_layer_index(ds)) if "mesh2d_nLayers" in da.dims else da


# ========= COMPUTE (heavy) =========
def compute_results(
    ds_path: str,
    out_dir: str,
    results_name: str = "results_minmax_surface.nc",
    donor_path: str | None = None,
    engine: str = "netcdf4",
    height_names=SURFACE_HEIGHT_NAMES,
    sal3d_names=SALINITY_3D_NAMES,
    sal2d_names=SALINITY_2D_NAMES,
    height_spinup_days: int = HEIGHT_SPINUP_DAYS,
    sal_spinup_days: int = SALINITY_SPINUP_DAYS,
    height_window_h: int = HEIGHT_WINDOW_H,
    sal_window_h: int = SALINITY_WINDOW_H,
) -> Path:
    """
    Repair UGRID, slice spin-up, take surface-only salinity,
    compute rolling extrema means, and write a compact results file.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / results_name

    print(f"[COMPUTE] Open: {ds_path}")
    ds = xr.open_dataset(ds_path, engine=engine)

    ds = _sanitize_all_topologies(ds, donor_path=donor_path)

    # variables
    src = ds
    sh = next((src[n] for n in height_names if n in src), None)
    if sh is None:
        raise KeyError(f"None of {height_names} found.")
    print(f"[COMPUTE] Using surface height: {sh.name} dims={sh.dims}")

    sal = None
    for n in sal3d_names:
        if n in src and "mesh2d_nLayers" in src[n].dims:
            sal = src[n]; print(f"[COMPUTE] Using salinity 3D: {sal.name} dims={sal.dims}"); break
    if sal is None:
        for n in sal2d_names:
            if n in src:
                sal = src[n]; print(f"[COMPUTE] Using salinity 2D: {sal.name} dims={sal.dims}"); break
    if sal is None:
        raise KeyError(f"None of {sal3d_names + sal2d_names} found.")

    # spin-up & surface
    sh = _slice_spinup(sh, height_spinup_days).astype("float32")
    sal_surf = _force_surface(sal, src)
    sal_surf = _slice_spinup(sal_surf, sal_spinup_days).astype("float32")

    # rolling -> means
    sh_min = _rolling_extreme_mean(sh, height_window_h, "min")
    sh_max = _rolling_extreme_mean(sh, height_window_h, "max")
    sa_min = _rolling_extreme_mean(sal_surf, sal_window_h, "min")
    sa_max = _rolling_extreme_mean(sal_surf, sal_window_h, "max")

    # reduce to 1-D over faces
    def _to_faces_1d(da):
        arr = da
        if "time" in arr.dims:
            arr = arr.mean("time")
        for d in list(arr.dims):
            if "face" not in d and d != "mesh2d_nFaces":
                arr = arr.mean(d)
        return arr

    sh_min = _to_faces_1d(sh_min).astype("float32")
    sh_max = _to_faces_1d(sh_max).astype("float32")
    sa_min = _to_faces_1d(sa_min).astype("float32")
    sa_max = _to_faces_1d(sa_max).astype("float32")

    # Stats + recommended height clim
    hvals = np.concatenate([sh_min.values.ravel(), sh_max.values.ravel()])
    hmin, hmax = np.nanmin(hvals), np.nanmax(hvals)
    hmag = float(np.nanmax(np.abs([hmin, hmax])))
    print(f"[STATS] Height MIN across fields: {hmin:.3f}, MAX: {hmax:.3f}")
    print(f"[STATS] Suggested symmetric height clim: (-{hmag:.2f}, {hmag:.2f})")

    svals = np.concatenate([sa_min.values.ravel(), sa_max.values.ravel()])
    smin, smax = np.nanmin(svals), np.nanmax(svals)
    print(f"[STATS] Salinity MIN across fields: {smin:.3f}, MAX: {smax:.3f} (plot will use 0–35)")

    # minimal mesh for plotting
    topo_name = _list_mesh_topologies(ds)[0]
    topo = ds[topo_name]
    face_nodes_name = topo.attrs.get("face_node_connectivity") or ("mesh2d_face_nodes" if "mesh2d_face_nodes" in ds else None)
    node_x_name = next((n for n in ("mesh2d_node_x", "node_x", "mesh2d_node_lon") if n in ds), None)
    node_y_name = next((n for n in ("mesh2d_node_y", "node_y", "mesh2d_node_lat") if n in ds), None)
    if not (face_nodes_name and node_x_name and node_y_name):
        raise RuntimeError("Could not find required mesh variables.")

    out = xr.Dataset(
        data_vars=dict(
            sh_min_mean=sh_min,
            sh_max_mean=sh_max,
            sa_min_mean=sa_min,
            sa_max_mean=sa_max,
            mesh2d_node_x=ds[node_x_name].astype("float64"),
            mesh2d_node_y=ds[node_y_name].astype("float64"),
            mesh2d_face_nodes=ds[face_nodes_name].astype("int32"),
        )
    )
    out["mesh2d"] = xr.DataArray(0, attrs=dict(
        cf_role="mesh_topology",
        topology_dimension=2,
        node_coordinates="mesh2d_node_x mesh2d_node_y",
        face_node_connectivity="mesh2d_face_nodes",
    ))
    out["mesh2d_face_nodes"].attrs["start_index"] = 0
    out["mesh2d_face_nodes"].encoding["_FillValue"] = -1

    enc = {k: dict(zlib=True, complevel=4) for k in out.data_vars}
    print(f"[COMPUTE] Write results: {results_path}")
    out.to_netcdf(results_path, engine="netcdf4", encoding=enc)
    print("[COMPUTE] Done.")
    return results_path


# ========= PLOT (fast) =========
def _tighten_axes(ax, node_x, node_y, xlim=None, ylim=None, pad=0.01):
    """Set tight limits from data or user-specified x/y limits. Add small padding."""
    if xlim is None:
        xmin, xmax = np.nanmin(node_x), np.nanmax(node_x)
    else:
        xmin = xmin if (xmin:=xlim[0]) is not None else np.nanmin(node_x)
        xmax = xmax if (xmax:=xlim[1]) is not None else np.nanmax(node_x)
    if ylim is None:
        ymin, ymax = np.nanmin(node_y), np.nanmax(node_y)
    else:
        ymin = ymin if (ymin:=ylim[0]) is not None else np.nanmin(node_y)
        ymax = ymax if (ymax:=ylim[1]) is not None else np.nanmax(node_y)

    dx = xmax - xmin; dy = ymax - ymin
    ax.set_xlim(xmin - pad*dx, xmax + pad*dx)
    ax.set_ylim(ymin - pad*dy, ymax + pad*dy)

def _poly_plot(face_nodes, node_x, node_y, values_1d, out_png, clim, title,
               stride=1, cmap=None, xlim=None, ylim=None, cbar_label=""):
    nfaces = face_nodes.shape[0]
    polys, good = [], []
    for i in range(0, nfaces, max(1, stride)):
        idxs = face_nodes[i]
        idxs = idxs[idxs >= 0]
        if idxs.size < 3:
            continue
        xs = node_x[idxs]; ys = node_y[idxs]
        if np.any(~np.isfinite(xs)) or np.any(~np.isfinite(ys)):
            continue
        polys.append(np.column_stack([xs, ys]))
        good.append(i)

    fig, ax = plt.subplots(figsize=(5.5, 7))
    ax.set_facecolor("lightgrey")

    coll = PolyCollection(polys, array=np.asarray(values_1d)[good], cmap=cmap)
    coll.set_rasterized(True)
    if clim is not None:
        coll.set_clim(*clim)
    ax.add_collection(coll)
    _tighten_axes(ax, node_x, node_y, xlim=xlim, ylim=ylim)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    # ax.set_title(title)
    cbar = fig.colorbar(coll, ax=ax)
    if cbar_label:
        cbar.set_label(cbar_label)

    plt.tight_layout(); plt.savefig(out_png, dpi=500)
    # plt.close(fig)

def plot_results(
    results_path: Path,
    out_dir: str,
    height_clim=HEIGHT_CLIM,
    salinity_clim=SALINITY_CLIM,
    poly_stride: int = POLY_STRIDE,
    x_limits=None,
    y_limits=None,
):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ds = xr.open_dataset(results_path, engine="netcdf4")

    # Choose colormaps
    cmap_height = cmap_topo
    # Salinity: yellow (fresh) -> blue (salty)
    # cmap_sal already defined above as cmo.cm.haline_r (blue=salty; reverse if you prefer)

    # Print min/max so you can standardise clims across runs
    def _mm(name):
        v = ds[name].values
        return float(np.nanmin(v)), float(np.nanmax(v))

    hmin1, hmax1 = _mm("sh_min_mean")
    hmin2, hmax2 = _mm("sh_max_mean")
    smin1, smax1 = _mm("sa_min_mean")
    smin2, smax2 = _mm("sa_max_mean")
    print(f"[STATS] sh_min_mean: min={hmin1:.3f}, max={hmax1:.3f}")
    print(f"[STATS] sh_max_mean: min={hmin2:.3f}, max={hmax2:.3f}")
    print(f"[STATS] sa_min_mean: min={smin1:.3f}, max={smax1:.3f}")
    print(f"[STATS] sa_max_mean: min={smin2:.3f}, max={smax2:.3f}")
    # Suggest symmetric height clim if user left None
    if height_clim is None:
        H = np.nanmax(np.abs([hmin1, hmax1, hmin2, hmax2]))
        height_clim = (-float(H), float(H))
        print(f"[INFO] Using suggested symmetric height clim: {height_clim}")

    face_nodes = np.asarray(ds["mesh2d_face_nodes"].data).astype("int32", copy=False)
    node_x = np.asarray(ds["mesh2d_node_x"].data).astype("float32", copy=False)
    node_y = np.asarray(ds["mesh2d_node_y"].data).astype("float32", copy=False)

    if _HAS_XUGRID:
        try:
            uds = xu.UgridDataset(ds)
            def _plot_xug(var, png, clim, title, cmap, cbar_label):
                fig, ax = plt.subplots(figsize=(6.8, 7.6))
                ax.set_facecolor("lightgrey")
                fig.patch.set_facecolor("lightgrey")
                uds[var].ugrid.plot(
                    ax=ax, cmap=cmap, vmin=clim[0], vmax=clim[1],
                    cbar_kwargs={"label": cbar_label}
                )
                _tighten_axes(ax, node_x, node_y, xlim=x_limits, ylim=y_limits)
                ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
                ax.set_title(title)
                plt.tight_layout(); plt.savefig(png, dpi=300)
                # plt.close(fig)

            _plot_xug("sh_min_mean", out_dir / "unstruct_surface_height_rollingMinMean_3h.png",
                      height_clim, "Surface height: mean rolling 3h MIN", cmap_height,
                      cbar_label="Surface Height [m]")
            _plot_xug("sh_max_mean", out_dir / "unstruct_surface_height_rollingMaxMean_3h.png",
                      height_clim, "Surface height: mean rolling 3h MAX", cmap_height,
                      cbar_label="Surface Height [m]")
            _plot_xug("sa_min_mean", out_dir / "unstruct_surface_salinity_rollingMinMean_24h.png",
                      salinity_clim, "Surface salinity: mean rolling 24h MIN", cmap_sal,
                      cbar_label="Salinity [psu]")
            _plot_xug("sa_max_mean", out_dir / "unstruct_surface_salinity_rollingMaxMean_24h.png",
                      salinity_clim, "Surface salinity: mean rolling 24h MAX", cmap_sal,
                      cbar_label="Salinity [psu]")
            print("[PLOT] Done (xugrid).")
            return
        except Exception as e:
            print(f"[WARN] xugrid failed ({e}); using PolyCollection fallback.")

    # Poly fallback (guaranteed 1-D colors)
    _poly_plot(face_nodes, node_x, node_y, ds["sh_min_mean"].values,
               out_dir / "unstruct_surface_height_rollingMinMean_3h.png",
               clim=height_clim, title="Surface height: mean rolling 3h MIN",
               stride=poly_stride, cmap=cmap_height, xlim=x_limits, ylim=y_limits,
               cbar_label="Surface Height [m]")
    _poly_plot(face_nodes, node_x, node_y, ds["sh_max_mean"].values,
               out_dir / "unstruct_surface_height_rollingMaxMean_3h.png",
               clim=height_clim, title="Surface height: mean rolling 3h MAX",
               stride=poly_stride, cmap=cmap_height, xlim=x_limits, ylim=y_limits,
               cbar_label="Surface Height [m]")
    _poly_plot(face_nodes, node_x, node_y, ds["sa_min_mean"].values,
               out_dir / "unstruct_surface_salinity_rollingMinMean_24h.png",
               clim=salinity_clim, title="Surface salinity: mean rolling 24h MIN",
               stride=poly_stride, cmap=cmap_sal, xlim=x_limits, ylim=y_limits,
               cbar_label="Salinity [psu]")
    _poly_plot(face_nodes, node_x, node_y, ds["sa_max_mean"].values,
               out_dir / "unstruct_surface_salinity_rollingMaxMean_24h.png",
               clim=salinity_clim, title="Surface salinity: mean rolling 24h MAX",
               stride=poly_stride, cmap=cmap_sal, xlim=x_limits, ylim=y_limits,
               cbar_label="Salinity [psu]")
    print("[PLOT] Done (PolyCollection).")


# ========= Spyder-friendly runner =========
if __name__ == "__main__":
    RECOMPUTE = False  # <- set True to force recompute
    RUN_COMPUTE = True
    RUN_PLOT    = True


    # ========= User defaults (edit) =========
    # Define all dataset/output pairs in one place
    runs = [
        ("/Volumes/PNB_extra/scw_10layer_realriv/kent_31_merged_map.nc",
         "/Volumes/PNB_extra/scw_10layer_realriv/exports_singlepoint"),
    
        ("/Volumes/PNB_extra/scw_10_layer_climatology/kent_31_merged_map.nc",
         "/Volumes/PNB_extra/scw_10_layer_climatology/exports_singlepoint"),
    
        ("/Volumes/PNB_extra/runSCW_ao_yawind_orig8RealRiver_m0.035_Forcing_85_Discouv-58757607/kent_31_merged_map.nc",
         "/Volumes/PNB_extra/runSCW_ao_yawind_orig8RealRiver_m0.035_Forcing_85_Discouv-58757607/exports_singlepoint"),
    
        ("/Volumes/PNB_extra/runSCW_ao_yawind_AllRivNoDuddonClimatology_m0.035_Forcing_85_Discouv-58757598/kent_31_merged_map.nc",
         "/Volumes/PNB_extra/runSCW_ao_yawind_AllRivNoDuddonClimatology_m0.035_Forcing_85_Discouv-58757598/exports_singlepoint")
    ]

    # Loop through each dataset/output directory pair
    for DS_PATH, OUT_DIR in runs:
        print(f"Processing dataset: {DS_PATH}")
        print(f"Output directory:   {OUT_DIR}")
    
        # === Your processing code goes here ===
        # e.g., load dataset, extract single-point time series, save outputs
        # ds = xr.open_dataset(DS_PATH)
        # process_and_save(ds, OUT_DIR)
    
        out_dir = Path(OUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)
        results_path = out_dir / RESULTS_NAME
    
        if RUN_COMPUTE:
            if results_path.exists() and not RECOMPUTE:
                print(f"[SKIP] Results already exist: {results_path}")
            else:
                results_path = compute_results(
                    ds_path=DS_PATH,
                    out_dir=OUT_DIR,
                    results_name=RESULTS_NAME,
                    donor_path=DONOR_PATH,
                    engine=ENGINE,
                    height_spinup_days=HEIGHT_SPINUP_DAYS,
                    sal_spinup_days=SALINITY_SPINUP_DAYS,
                    height_window_h=HEIGHT_WINDOW_H,
                    sal_window_h=SALINITY_WINDOW_H,
                )
    
        if RUN_PLOT:
            plot_results(
                results_path=results_path,
                out_dir=OUT_DIR,
                height_clim=HEIGHT_CLIM,     # set to None to auto symmetric based on data
                salinity_clim=SALINITY_CLIM, # fixed 0–35
                poly_stride=POLY_STRIDE,
                x_limits=X_LIMITS,
                y_limits=Y_LIMITS,
            )
