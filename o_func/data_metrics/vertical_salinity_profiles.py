# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Single-point vertical salinity profiles using ONLY water depth H(t).
# - Evenly split the water column into N layers each time step.
# - Interpolate salinity to those N mid-depths (normalized 0..1).
# - Plot a 3×4 grid with global (shared) axes limits and global axis labels.

# Expected variables in the D-Flow FM NetCDF:
# - Face centers: mesh2d_face_x, mesh2d_face_y
# - Water depth (time × faces): one of
#     ['mesh2d_waterdepth','waterdepth','water_depth','mesh2d_h','h',
#      'mesh2d_water_depth','mesh2d_flowelem_hp','mesh2d_hp']
# - Salinity: any var with dims including ('time','mesh2d_nFaces', vertical dim)
# """

# from pathlib import Path
# import numpy as np
# import xarray as xr
# import matplotlib.pyplot as plt

# # ------------------ USER INPUTS ------------------
# dataset = '/Volumes/PNB_extra/scw_10layer_realriv/kent_31_merged_map.nc'

# # Location
# target_lat = 53.31093730977143
# target_lon = -3.1802285735308873

# # Time aggregation ('native' for none; else Pandas offset alias like '1H','20min','1D',...)
# time_resample = '1H'

# # Grid (consecutive timesteps on the resampled series)
# grid_n_panels = 12
# grid_step = 1
# grid_start_index = 1400   # or set None and use grid_start_iso
# grid_start_iso   = None    # e.g. "2013-11-19 12:00"

# # Vertical layers (evenly split)
# n_layers_target = 10

# # Figure (A4-ish portrait; you asked for 8x11)
# figsize_grid = (8, 11)
# dpi_out = 200

# # Outputs
# out_dir = Path('/Volumes/PNB_extra/scw_10layer_realriv/exports_singlepoint')
# out_dir.mkdir(parents=True, exist_ok=True)
# profile_grid_png = out_dir / f'salinity_profile_grid_evenDepth_{time_resample}.png'
# # -------------------------------------------------

# # ------------------ Helpers ------------------
# def haversine(lon1, lat1, lon2, lat2):
#     R = 6371000.0
#     phi1, phi2 = np.radians(lat1), np.radians(lat2)
#     dphi = phi2 - phi1
#     dlambda = np.radians(lon2 - lon1)
#     a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2.0)**2
#     return 2 * R * np.arcsin(np.sqrt(a))

# def pick_first(ds, names):
#     for n in names:
#         if n in ds.variables:
#             return n
#     return None

# def choose_salinity_var(ds):
#     cand = list(ds.data_vars)
#     def looks_like(n):
#         n_ = n.lower()
#         return ('salin' in n_) or (n_ in ('mesh2d_sa1','sa1','salinity','mesh2d_salinity'))
#     sal = [n for n in cand if looks_like(n)]
#     if not sal:
#         for n in cand:
#             d = ds[n].dims
#             if ('time' in d) and ('mesh2d_nFaces' in d) and (('mesh2d_nLayers' in d) or ('mesh2d_nInterfaces' in d)):
#                 sal = [n]; break
#     if not sal:
#         raise KeyError("No salinity-like variable found.")
#     def has_z(n):
#         d = ds[n].dims
#         return ('mesh2d_nLayers' in d) or ('mesh2d_nInterfaces' in d)
#     return sorted(sal, key=lambda n: (not has_z(n), n))[0]

# def get_vertical_dim(da):
#     d = da.dims
#     if 'mesh2d_nLayers' in d: return 'mesh2d_nLayers'
#     if 'mesh2d_nInterfaces' in d: return 'mesh2d_nInterfaces'
#     raise ValueError("Salinity variable has no recognized vertical dimension.")

# def maybe_resample(da, rule):
#     if rule is None: return da
#     return da.resample(time=rule, label='left').mean()

# # Interpolate salinity along normalized vertical (0..1) to N target points
# def remap_to_uniform_layers(S_t, s_orig_norm, n_target):
#     if S_t.size == n_target:
#         return S_t
#     s_target = (np.arange(n_target) + 0.5) / n_target
#     return np.interp(s_target, s_orig_norm, S_t)

# # ------------------ Open and select face ------------------
# time_chunk = 96
# ds = xr.open_dataset(dataset, chunks={'time': time_chunk})

# # Face centers
# for needed in ['mesh2d_face_x','mesh2d_face_y']:
#     if needed not in ds.variables:
#         raise KeyError(f"Missing coordinate variable: {needed}")
# face_lon = ds['mesh2d_face_x'].values
# face_lat = ds['mesh2d_face_y'].values

# # Nearest face
# dists = haversine(face_lon, face_lat, target_lon, target_lat)
# face_idx = int(np.argmin(dists))
# print(f"Nearest face index = {face_idx} (≈ {dists[face_idx]:.1f} m)")

# # ------------------ Variables ------------------
# # Salinity
# sal_var = choose_salinity_var(ds)
# zdim = get_vertical_dim(ds[sal_var])
# sal_face = ds[sal_var].isel(mesh2d_nFaces=face_idx)   # [time, z]
# nz_orig = sal_face.sizes[zdim]

# # Water depth H(t)
# wd_candidates = [
#     'mesh2d_waterdepth','waterdepth','water_depth','mesh2d_h',
#     'h','mesh2d_water_depth','mesh2d_flowelem_hp','mesh2d_hp'
# ]
# wd_name = pick_first(ds, wd_candidates)
# if wd_name is None:
#     raise KeyError("Could not find a water depth variable. Tried: " + ", ".join(wd_candidates))
# H = ds[wd_name].isel(mesh2d_nFaces=face_idx)  # [time] or scalar (rare)
# if 'time' not in H.dims:
#     # broadcast to salinity time axis
#     H = xr.DataArray(np.broadcast_to(H.values, sal_face.sizes['time']), dims=['time'])

# # ------------------ Resample (if requested) ------------------
# rule = None if (time_resample is None or str(time_resample).lower() == 'native') else str(time_resample)
# sal_rs = maybe_resample(sal_face, rule)   # [time_rs, z]
# H_rs   = maybe_resample(H, rule)          # [time_rs]

# # ------------------ Build evenly spaced depths (positive down) ------------------
# N = int(n_layers_target)
# s_target = (np.arange(N) + 0.5) / N  # midpoints in [0,1]
# # Depth below surface (m, positive down): d_k(t) = ((k+0.5)/N) * H(t)
# depth_even = H_rs.expand_dims({'lev': N}) * xr.DataArray(s_target, dims=['lev'])
# depth_even = depth_even.transpose('time', 'lev')  # [time, lev]
# ylab = 'Depth below surface [m]'

# # ------------------ Panel time indices ------------------
# times_all = sal_rs['time'].values
# n_time = times_all.shape[0]

# # Clamp start index so we can fill panels if possible
# if grid_start_index is not None:
#     max_start = max(0, n_time - grid_step*(grid_n_panels-1) - 1)
#     start_idx = min(max(0, grid_start_index), max_start)
# elif grid_start_iso is not None:
#     try:
#         start_idx = int(np.argmin(np.abs(times_all - np.datetime64(grid_start_iso))))
#         start_idx = min(start_idx, max(0, n_time - grid_step*(grid_n_panels-1) - 1))
#     except Exception:
#         start_idx = 0
# else:
#     start_idx = 0

# idxs_grid = []
# for k in range(grid_n_panels):
#     i = start_idx + k * grid_step
#     if i < n_time:
#         idxs_grid.append(i)
# if not idxs_grid:
#     raise ValueError("No valid indices for grid after (re)sampling.")

# # ------------------ Interpolate salinity to N uniform layers ------------------
# # Build normalized coordinate for the *original* vertical positions (0..1).
# # If sigma arrays exist, use them; else assume uniform.
# if 'mesh2d_layer_sigma' in ds.variables and zdim == 'mesh2d_nLayers':
#     s_orig = xr.DataArray(ds['mesh2d_layer_sigma'].values, dims=[zdim])
# elif 'mesh2d_interface_sigma' in ds.variables and zdim == 'mesh2d_nInterfaces':
#     si = ds['mesh2d_interface_sigma'].values
#     # we’ll use interfaces directly (0..1)
#     s_orig = xr.DataArray(si, dims=[zdim])
# else:
#     if zdim == 'mesh2d_nLayers':
#         s_orig = xr.DataArray((np.arange(nz_orig) + 0.5) / nz_orig, dims=[zdim])
#     else:
#         s_orig = xr.DataArray(np.linspace(0, 1, nz_orig), dims=[zdim])
# s_orig_np = np.asarray(s_orig.values, dtype=float)

# # Extract salinity and map to N targets for selected times
# sal_sel = sal_rs.isel(time=idxs_grid).load().values  # [npanels, nz_orig]
# S_even = np.empty((len(idxs_grid), N), dtype=float)
# for p in range(len(idxs_grid)):
#     S_even[p, :] = remap_to_uniform_layers(sal_sel[p, :], s_orig_np, N)

# # Depth array for the same times (positive down)
# D_even = depth_even.isel(time=idxs_grid).load().values  # [npanels, N]

# # ------------------ Axis limits (consistent across panels) ------------------
# # X limits from salinity across selected times
# x_min_g, x_max_g = np.nanpercentile(S_even, [1, 99])
# pad_g = max(0.5, 0.1*(x_max_g - x_min_g))
# xlim_g = (x_min_g - pad_g, x_max_g + pad_g)

# # Y limits: 0 (surface) to max depth across selected times
# H_sel = H_rs.isel(time=idxs_grid).load().values
# y_top = 0.0
# y_bottom = float(np.nanmax(H_sel))  # deepest water among panels
# ylim_g = (y_bottom, y_top)  # deepest at bottom, surface at top

# print(f"Grid xlim: {xlim_g}, depth range across panels: 0..{y_bottom:.2f} m")

# # ------------------ Plot 3×4 grid ------------------
# ncols, nrows = 3, 4
# letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# # Use constrained_layout so global labels sit outside nicely
# fig, axes = plt.subplots(nrows, ncols, figsize=figsize_grid, sharex=True, sharey=True,
#                          constrained_layout=True)
# axes_flat = axes.ravel()

# for p, ax in enumerate(axes_flat):
#     if p < len(idxs_grid):
#         s = S_even[p, :]
#         d = D_even[p, :]
#         ax.plot(s, d, lw=2)
#         ax.set_xlim(xlim_g); ax.set_ylim(ylim_g)
#         ts = np.datetime_as_string(times_all[idxs_grid[p]], unit='m')
#         ax.set_title(f"{letters[p]} ({ts})", fontsize=9)
#         ax.grid(True, alpha=0.25)
#     else:
#         ax.axis('off')

# # One combined x/y label outside the panels
# fig.supxlabel('Salinity [psu]')
# fig.supylabel(ylab)

# fig.savefig(profile_grid_png, dpi=dpi_out)
# print(f"Saved 3×4 grid to: {profile_grid_png}")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch single-point vertical salinity profiles using ONLY water depth H(t).

- For each (dataset, out_dir) in `runs`, select nearest face to (target_lon, target_lat)
- Evenly split the water column into N layers at each selected time
- Interpolate salinity to those N mid-depths (normalized 0..1)
- Plot a 3×4 grid with shared/global axis limits and global axis labels
- Save to the provided out_dir (created if missing)
"""

from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# ------------------ USER INPUTS ------------------
# Location (nearest face will be selected per file)
target_lat = 53.31093730977143
target_lon = -3.1802285735308873

# Time aggregation ('native' for none; else Pandas offset alias like '1H','20min','1D',...)
time_resample = '1H'

# Grid (consecutive timesteps on the resampled series)
grid_n_panels = 12
grid_step = 1
grid_start_index = 1400   # or set None and use grid_start_iso
grid_start_iso   = None    # e.g. "2013-11-19 12:00"

# Vertical layers (evenly split)
n_layers_target = 10

# Figure (A4-ish portrait; you asked for 8×11)
figsize_grid = (8, 11)
dpi_out = 200

# Define all dataset/output pairs in one place
runs = [
    ("/Volumes/PNB_extra/scw_10layer_realriv/kent_31_merged_map.nc",
     "/Volumes/PNB_extra/scw_10layer_realriv/exports_singlepoint"),

    ("/Volumes/PNB_extra/scw_10_layer_climatology/kent_31_merged_map.nc",
     "/Volumes/PNB_extra/scw_10_layer_climatology/exports_singlepoint")
    ]

# ------------------ Helpers ------------------
def haversine(lon1, lat1, lon2, lat2):
    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def pick_first(ds, names):
    for n in names:
        if n in ds.variables:
            return n
    return None

def choose_salinity_var(ds):
    cand = list(ds.data_vars)
    def looks_like(n):
        n_ = n.lower()
        return ('salin' in n_) or (n_ in ('mesh2d_sa1','sa1','salinity','mesh2d_salinity'))
    sal = [n for n in cand if looks_like(n)]
    if not sal:
        # fallback: any var with (time, faces, vertical) dims
        for n in cand:
            d = ds[n].dims
            if ('time' in d) and ('mesh2d_nFaces' in d) and (('mesh2d_nLayers' in d) or ('mesh2d_nInterfaces' in d)):
                sal = [n]; break
    if not sal:
        raise KeyError("No salinity-like variable found.")
    def has_z(n):
        d = ds[n].dims
        return ('mesh2d_nLayers' in d) or ('mesh2d_nInterfaces' in d)
    return sorted(sal, key=lambda n: (not has_z(n), n))[0]

def get_vertical_dim(da):
    d = da.dims
    if 'mesh2d_nLayers' in d: return 'mesh2d_nLayers'
    if 'mesh2d_nInterfaces' in d: return 'mesh2d_nInterfaces'
    raise ValueError("Salinity variable has no recognized vertical dimension.")

def maybe_resample(da, rule):
    if rule is None: return da
    return da.resample(time=rule, label='left').mean()

# Interpolate salinity along normalized vertical (0..1) to N target points
def remap_to_uniform_layers(S_t, s_orig_norm, n_target):
    if S_t.size == n_target:
        return S_t
    s_target = (np.arange(n_target) + 0.5) / n_target
    return np.interp(s_target, s_orig_norm, S_t)

def process_one_file(dataset_path: str, out_dir: Path):
    dataset = str(dataset_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Processing: {dataset} ===")

    time_chunk = 96
    with xr.open_dataset(dataset, chunks={'time': time_chunk}) as ds:
        # Face centers
        for needed in ['mesh2d_face_x','mesh2d_face_y']:
            if needed not in ds.variables:
                raise KeyError(f"Missing coordinate variable: {needed}")
        face_lon = ds['mesh2d_face_x'].values
        face_lat = ds['mesh2d_face_y'].values

        # Nearest face
        dists = haversine(face_lon, face_lat, target_lon, target_lat)
        face_idx = int(np.argmin(dists))
        print(f"Nearest face index = {face_idx} (≈ {dists[face_idx]:.1f} m)")

        # Salinity
        sal_var = choose_salinity_var(ds)
        zdim = get_vertical_dim(ds[sal_var])
        sal_face = ds[sal_var].isel(mesh2d_nFaces=face_idx)   # [time, z]
        nz_orig = sal_face.sizes[zdim]

        # Water depth H(t)
        wd_candidates = [
            'mesh2d_waterdepth','waterdepth','water_depth','mesh2d_h',
            'h','mesh2d_water_depth','mesh2d_flowelem_hp','mesh2d_hp'
        ]
        wd_name = pick_first(ds, wd_candidates)
        if wd_name is None:
            raise KeyError("Could not find a water depth variable. Tried: " + ", ".join(wd_candidates))
        H = ds[wd_name].isel(mesh2d_nFaces=face_idx)  # [time] or scalar (rare)
        if 'time' not in H.dims:
            # broadcast to salinity time axis
            H = xr.DataArray(np.broadcast_to(np.asarray(H.values, dtype=float), (sal_face.sizes['time'],)),
                             dims=['time'])

        # Resample (if requested)
        rule = None if (time_resample is None or str(time_resample).lower() == 'native') else str(time_resample)
        sal_rs = maybe_resample(sal_face, rule)   # [time_rs, z]
        H_rs   = maybe_resample(H, rule)          # [time_rs]

        # Build evenly spaced depths (positive down) — d_k(t) = ((k+0.5)/N) * H(t)
        N = int(n_layers_target)
        s_target = (np.arange(N) + 0.5) / N  # midpoints in [0,1]
        depth_even = H_rs.expand_dims({'lev': N}) * xr.DataArray(s_target, dims=['lev'])
        depth_even = depth_even.transpose('time', 'lev')  # [time, lev]
        ylab = 'Depth below surface [m]'

        # Panel time indices
        times_all = sal_rs['time'].values
        n_time = times_all.shape[0]
        if n_time == 0:
            raise ValueError("No time steps after (re)sampling.")

        if grid_start_index is not None:
            max_start = max(0, n_time - grid_step*(grid_n_panels-1) - 1)
            start_idx = min(max(0, grid_start_index), max_start)
        elif grid_start_iso is not None:
            try:
                start_idx = int(np.argmin(np.abs(times_all - np.datetime64(grid_start_iso))))
                start_idx = min(start_idx, max(0, n_time - grid_step*(grid_n_panels-1) - 1))
            except Exception:
                start_idx = 0
        else:
            start_idx = 0

        idxs_grid = []
        for k in range(grid_n_panels):
            i = start_idx + k * grid_step
            if i < n_time:
                idxs_grid.append(i)
        if not idxs_grid:
            raise ValueError("No valid indices for grid after (re)sampling.")

        # Build normalized coordinate for the *original* vertical positions (0..1).
        # If sigma arrays exist, use them; else assume uniform.
        if 'mesh2d_layer_sigma' in ds.variables and zdim == 'mesh2d_nLayers':
            s_orig = xr.DataArray(ds['mesh2d_layer_sigma'].values, dims=[zdim])
        elif 'mesh2d_interface_sigma' in ds.variables and zdim == 'mesh2d_nInterfaces':
            si = ds['mesh2d_interface_sigma'].values
            s_orig = xr.DataArray(si, dims=[zdim])
        else:
            if zdim == 'mesh2d_nLayers':
                s_orig = xr.DataArray((np.arange(nz_orig) + 0.5) / nz_orig, dims=[zdim])
            else:
                s_orig = xr.DataArray(np.linspace(0, 1, nz_orig), dims=[zdim])
        s_orig_np = np.asarray(s_orig.values, dtype=float)

        # Extract salinity and map to N targets for selected times
        sal_sel = sal_rs.isel(time=idxs_grid).load().values  # [npanels, nz_orig]
        S_even = np.empty((len(idxs_grid), N), dtype=float)
        for p in range(len(idxs_grid)):
            S_even[p, :] = remap_to_uniform_layers(sal_sel[p, :], s_orig_np, N)

        # Depth array for the same times (positive down)
        D_even = depth_even.isel(time=idxs_grid).load().values  # [npanels, N]

        # Axis limits (consistent across panels)
        x_min_g, x_max_g = np.nanpercentile(S_even, [1, 99])
        pad_g = max(0.5, 0.1*(x_max_g - x_min_g))
        xlim_g = (x_min_g - pad_g, x_max_g + pad_g)

        H_sel = H_rs.isel(time=idxs_grid).load().values
        y_top = 0.0
        y_bottom = float(np.nanmax(H_sel))
        ylim_g = (y_bottom, y_top)

        print(f"Grid xlim: {xlim_g}, depth range across panels: 0..{y_bottom:.2f} m")

        # Plot 3×4 grid
        ncols, nrows = 3, 4
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize_grid, sharex=True, sharey=True,
                                 constrained_layout=True)
        axes_flat = axes.ravel()

        for p, ax in enumerate(axes_flat):
            if p < len(idxs_grid):
                s = S_even[p, :]
                d = D_even[p, :]
                ax.plot(s, d, lw=2)
                ax.set_xlim(xlim_g); ax.set_ylim(ylim_g)
                ts = np.datetime_as_string(times_all[idxs_grid[p]], unit='m')
                ax.set_title(f"{letters[p]} ({ts})", fontsize=9)
                ax.grid(True, alpha=0.25)
            else:
                ax.axis('off')

        # Global labels
        fig.supxlabel('Salinity [psu]')
        fig.supylabel(ylab)

        # Output filename
        ds_stem = Path(dataset).stem  # "kent_31_merged_map"
        res_tag = 'native' if (rule is None) else rule
        out_png = out_dir / f"{ds_stem}_salinity_profile_grid_evenDepth_{res_tag}.png"

        fig.savefig(out_png, dpi=dpi_out)
        plt.close(fig)
        print(f"✔ Saved 3×4 grid to: {out_png}")

# ------------------ Run batch ------------------
if __name__ == "__main__":
    n_ok, n_fail = 0, 0
    for ds_path, out_dir in runs:
        try:
            process_one_file(ds_path, Path(out_dir))
            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"✖ Failed: {ds_path}\n   Reason: {type(e).__name__}: {e}")
    print(f"\nDone. Successful: {n_ok}, Failed: {n_fail}")
