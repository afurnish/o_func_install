from pathlib import Path
from o_func import opsys


def plot_climatology_simple(save_path="."):
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    # Define paths using opsys
    start_path = Path(opsys('PN'))
    climatology_file_path = start_path / 'GitHub/o_func_install/o_func/data_prepkit/all_rivers.csv'
    rt = 'modelling_DATA/kent_estuary_project/river_boundary_conditions/delft_3dfm_inputs/001_river_data_2013-10-28_to_2014-03-01'
    other_riv = 'modelling_DATA/kent_estuary_project/river_boundary_conditions/delft_bc_files/extra_rivers_real_data_esk_alt_clywd/processed'
    river_path = start_path / Path(rt)
    other_river_path = start_path / Path(other_riv)

    # Load climatology data
    climatology_df = pd.read_csv(climatology_file_path, index_col=0, parse_dates=True)

    # Initialize dictionaries to store data
    riv_files = {}
    combined_data = {}

    # Collect and organize file paths for river data
    for file_riv in river_path.glob('*'):
        river_name = file_riv.name.split('_')[0].lower()
        riv_files.setdefault(river_name, []).append(file_riv)

    for file_riv in other_river_path.glob('*'):
        river_name = file_riv.name.split('_')[0].lower()
        riv_files.setdefault(river_name, []).append(file_riv)

    # Combine into a single dictionary
    all_river_names = sorted(riv_files.keys())
    for river_name in all_river_names:
        combined_data[river_name] = {
            'river_data': riv_files.get(river_name, []),
            'climatology_data': climatology_df[river_name.capitalize()] if river_name.capitalize() in climatology_df.columns else None
        }
    

    # Define plot dimensions
    num_rivers = len(combined_data)
    fig_width = 10  # Fixed width
    fig_height = num_rivers * 1.5  # Height dynamically scales with number of rivers

    # Create the figure
    fig, axes = plt.subplots(nrows=num_rivers, ncols=1, figsize=(fig_width, fig_height), sharex=True)
    if num_rivers == 1:
        axes = [axes]  # Ensure axes is iterable if there's only one subplot

    # Plot each dataset
    start_date = pd.Timestamp("2013-10-28 00:00:00")
    end_date = pd.Timestamp("2014-03-01 00:00:00")
    for ax, (river_name, data) in zip(axes, combined_data.items()):
        has_data = False
        if data['river_data']:
            for river_file in data['river_data']:
                river_data = pd.read_csv(river_file, header=None, names=["time", "value"])
                river_data['time'] = pd.to_datetime(river_data['time'])
                river_data = river_data[(river_data['time'] >= start_date) & (river_data['time'] <= end_date)]
                if not river_data.empty:
                    ax.plot(river_data['time'], river_data['value'], label='River Data', color='blue', alpha=0.7)
                    has_data = True

        if data['climatology_data'] is not None:
            climatology_series = data['climatology_data'][
                (climatology_df.index >= start_date) & (climatology_df.index <= end_date)]
            if not climatology_series.empty:
                ax.plot(climatology_series.index, climatology_series.values, label='Climatology Data', color='red', alpha=0.7)
                has_data = True

        # Annotate river name
        ax.text(0.02, 0.85, river_name.capitalize(), transform=ax.transAxes, ha='left', va='top',
                fontsize=12, fontweight='bold')

        # Hide the subplot if no data
        if not has_data:
            ax.axis('off')

        if river_name == 'mersey':
            fig2, axes2 = plt.subplots()
            axes2.plot(climatology_series.index, climatology_series.values)
            plt.savefig('Mersey.png')
            plt.close()
            
    # Add labels
    axes[-1].set_xlabel('Date', fontsize=14, labelpad=15)  # Increased font size and added padding
    fig.text(0.04, 0.5, 'Discharge (m³/s)', va='center', rotation='vertical', fontsize=14)  # Increased font size and padding

    # Rotate x-axis labels diagonally
    for ax in axes:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add legend closer to the figure
    fig.legend(['15-min NRFA measured river discharge', 'AMM15 river climatology discharge'], loc='upper center', ncol=2, fontsize=12, bbox_to_anchor=(0.5, 0.91))

    # Save the figure
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    figure_path = save_path / "climatology_vs_river_discharge.png"
    fig.savefig(figure_path, dpi=300, bbox_inches='tight')
    # plt.close(fig)

    print(f"Figure saved to {figure_path}")


if __name__ == '__main__':
    # Specify save path
    start_path = Path(opsys('PN'))
    savepath = start_path / Path('modelling_DATA/kent_estuary_project/river_boundary_conditions/figures')
    plot_climatology_simple(save_path=savepath)
    
    
#%% Compute stats on water discharge
# def plot_climatology(layout="landscape", num_figures=1, save_path=".", sharey=False,
#                      start_date="2013-10-28 00:00:00", end_date="2014-03-01 00:00:00",
#                      rivers_per_figure=None, write_cumulative_csv=True, write_all_combos=True):
#     """
#     Plots river vs climatology AND writes cumulative discharge CSVs (m^3) with columns:
#       - river
#       - cumulative_climatology_m3
#       - cumulative_real_m3
#       - cumulative_difference_m3         = climatology - real
#       - ratio_clim_over_real             = climatology / real
#       - relative_difference_to_real      = (climatology - real) / real

#     CSVs written in save_path:
#       - all_rivers.csv
#       - no_duddon.csv
#       - no_clywd_esk_alt.csv
#       - (optional) every exclusion combo of {duddon, clwyd/clywd, esk, alt} as exclude_*.csv
#     """
#     import re
#     from pathlib import Path
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     import numpy as np
#     from itertools import combinations
#     from o_func import opsys

#     # ------------ helpers ------------
#     def norm(name: str) -> str:
#         return re.sub(r'[^a-z0-9]+', '', str(name).lower())

#     def canonical(key: str) -> str:
#         # coalesce common variants (e.g., clywd -> clwyd)
#         key = norm(key)
#         return {'clywd': 'clwyd'}.get(key, key)

#     # NumPy trapezoid (deprecation-safe)
#     try:
#         from numpy import trapezoid as _trapz
#     except Exception:
#         from numpy import trapz as _trapz

#     def integrate_series(s: pd.Series) -> float:
#         """Numerical integral of discharge (m^3/s) over time window -> volume (m^3)."""
#         if s is None or s.empty:
#             return np.nan
#         s = s.dropna().sort_index()
#         if s.empty:
#             return np.nan
#         idx = s.index
#         # Handle possible timezone-awareness
#         if getattr(idx, 'tz', None) is not None:
#             idx = idx.tz_convert('UTC').tz_localize(None)
#         t = idx.astype('int64') / 1e9  # seconds
#         y = s.values
#         return float(_trapz(y, t))  # m^3

#     def read_real_series(file_paths, t0, t1) -> pd.Series:
#         """Read/merge any number of 'real' river files -> single time-indexed Series."""
#         frames = []
#         for p in file_paths or []:
#             df = pd.read_csv(p, header=None, names=["time", "value"])
#             df['time'] = pd.to_datetime(df['time'])
#             df = df[(df['time'] >= t0) & (df['time'] <= t1)]
#             if not df.empty:
#                 frames.append(df[['time', 'value']].set_index('time').sort_index())
#         if not frames:
#             return pd.Series(dtype=float)
#         merged = pd.concat(frames).groupby(level=0).mean().sort_index()  # avoid double counting duplicates
#         return merged['value']

#     def safe_ratio(clim, real, eps=1e-9):
#         if pd.notna(clim) and pd.notna(real) and abs(real) > eps:
#             return clim / real
#         return np.nan

#     def safe_rel_diff(clim, real, eps=1e-9):
#         if pd.notna(clim) and pd.notna(real) and abs(real) > eps:
#             return (clim - real) / real
#         return np.nan

#     # ------------ paths ------------
#     start_path = Path(opsys('PN'))
#     climatology_file_path = start_path / 'GitHub/o_func_install/o_func/data_prepkit/all_rivers.csv'
#     rt = 'modelling_DATA/kent_estuary_project/river_boundary_conditions/delft_3dfm_inputs/001_river_data_2013-10-28_to_2014-03-01'
#     other_riv = 'modelling_DATA/kent_estuary_project/river_boundary_conditions/delft_bc_files/extra_rivers_real_data_esk_alt_clywd/processed'
#     river_path = start_path / rt
#     other_river_path = start_path / other_riv

#     # ------------ load climatology ------------
#     climatology_df = pd.read_csv(climatology_file_path, index_col=0, parse_dates=True)
#     clim_map = {canonical(c): climatology_df[c] for c in climatology_df.columns}
#     display_from_clim = {canonical(c): c for c in climatology_df.columns}

#     # ------------ scan river files ------------
#     riv_files = {}
#     display_from_files = {}
#     for p in list(river_path.glob('*')) + list(other_river_path.glob('*')):
#         if not p.is_file():
#             continue
#         base = p.name.split('_')[0]
#         key = canonical(base)
#         riv_files.setdefault(key, []).append(p)
#         display_from_files.setdefault(key, base.title())

#     # ------------ union of rivers ------------
#     all_keys = sorted(set(clim_map.keys()) | set(riv_files.keys()))
#     combined = {}
#     for key in all_keys:
#         disp = display_from_clim.get(key, display_from_files.get(key, key.title()))
#         combined[disp] = {
#             "key": key,
#             "river_data": riv_files.get(key, []),
#             "climatology_data": clim_map.get(key, None),
#         }

#     # ------------ time window & layout ------------
#     t0 = pd.Timestamp(start_date)
#     t1 = pd.Timestamp(end_date)
#     if layout == "landscape":
#         base_width, base_height = 14, 200
#     elif layout == "portrait":
#         base_width, base_height = 10, 250
#     else:
#         raise ValueError("Invalid layout choice. Use 'landscape' or 'portrait'.")

#     num_rivers = len(combined)
#     if rivers_per_figure is None:
#         rivers_per_figure = (num_rivers + num_figures - 1) // max(num_figures, 1)

#     # ------------ ensure save path ------------
#     save_path = Path(save_path)
#     save_path.mkdir(parents=True, exist_ok=True)

#     # ------------ compute cumulative volumes (m^3) ------------
#     rows = []
#     for disp, data in combined.items():
#         series_real = read_real_series(data["river_data"], t0, t1)
#         series_clim = None
#         if data["climatology_data"] is not None:
#             s = data["climatology_data"]
#             series_clim = s[(s.index >= t0) & (s.index <= t1)]
#         vol_real = integrate_series(series_real)
#         vol_clim = integrate_series(series_clim)
#         diff = (vol_clim - vol_real) if (pd.notna(vol_clim) or pd.notna(vol_real)) else np.nan
#         ratio = safe_ratio(vol_clim, vol_real)
#         rel = safe_rel_diff(vol_clim, vol_real)
#         rows.append({
#             "river": disp,
#             "key": data["key"],
#             "cumulative_climatology_m3": vol_clim,
#             "cumulative_real_m3": vol_real,
#             "cumulative_difference_m3": diff,              # climatology - real
#             "ratio_clim_over_real": ratio,                  # e.g., 3.5 means 3.5x
#             "relative_difference_to_real": rel,             # e.g., 0.2 means +20%
#         })
#     cum_df_all = pd.DataFrame(rows)

#     def write_summary(label: str, exclude_keys=None):
#         exclude_keys = set(canonical(x) for x in (exclude_keys or []))
#         df = cum_df_all[~cum_df_all["key"].isin(exclude_keys)].copy()

#         total_clim = df["cumulative_climatology_m3"].sum(skipna=True)
#         total_real = df["cumulative_real_m3"].sum(skipna=True)
#         total_diff = total_clim - total_real
#         total_ratio = safe_ratio(total_clim, total_real)
#         total_rel = safe_rel_diff(total_clim, total_real)

#         total = {
#             "river": "TOTAL",
#             "cumulative_climatology_m3": total_clim,
#             "cumulative_real_m3": total_real,
#             "cumulative_difference_m3": total_diff,
#             "ratio_clim_over_real": total_ratio,
#             "relative_difference_to_real": total_rel,
#         }
#         out = pd.concat(
#             [df[["river",
#                  "cumulative_climatology_m3",
#                  "cumulative_real_m3",
#                  "cumulative_difference_m3",
#                  "ratio_clim_over_real",
#                  "relative_difference_to_real"]],
#              pd.DataFrame([total])],
#             ignore_index=True
#         )
#         out_path = save_path / f"{label}.csv"
#         out.to_csv(out_path, index=False)
#         return out_path

#     if write_cumulative_csv:
#         # Required named outputs
#         write_summary("all_rivers", exclude_keys=[])
#         write_summary("no_duddon", exclude_keys=["duddon"])
#         write_summary("no_clywd_esk_alt", exclude_keys=["clwyd", "clywd", "esk", "alt"])  # handles both spellings

#         # Every combination of the set {duddon, clwyd, esk, alt}
#         if write_all_combos:
#             base_exclude_set = ["duddon", "clwyd", "esk", "alt"]
#             seen = set()
#             for r in range(1, len(base_exclude_set) + 1):
#                 for combo in combinations(base_exclude_set, r):
#                     canon_combo = tuple(sorted(canonical(x) for x in combo))
#                     if canon_combo in seen:
#                         continue
#                     seen.add(canon_combo)
#                     label = "exclude_" + "_".join(canon_combo)
#                     write_summary(label, exclude_keys=list(canon_combo))

#     # ------------ PLOTTING (unchanged except pagination fix) ------------
#     items = list(combined.items())
#     rivers_per_figure = max(1, rivers_per_figure)
#     total_figs = int(np.ceil(num_rivers / rivers_per_figure))

#     for fig_idx in range(total_figs):
#         start_idx = fig_idx * rivers_per_figure
#         end_idx = min((fig_idx + 1) * rivers_per_figure, num_rivers)
#         rivers_in_figure = items[start_idx:end_idx]
#         num_rows = len(rivers_in_figure)

#         fig, axes = plt.subplots(nrows=num_rows, ncols=1, figsize=(base_width, base_height),
#                                  sharex=True, sharey=sharey)
#         axes = axes.flatten() if num_rows > 1 else [axes]

#         all_legend_handles = []
#         for ax, (river_name, data) in zip(axes, rivers_in_figure):
#             has_data = False
#             legend_handles = []

#             # Real data
#             if data['river_data']:
#                 s_real = read_real_series(data['river_data'], t0, t1)
#                 if not s_real.empty:
#                     ln, = ax.plot(s_real.index, s_real.values, label='River Data', alpha=0.7)
#                     has_data = True
#                     legend_handles.append(ln)

#             # Climatology
#             s_clim = data['climatology_data']
#             if s_clim is not None:
#                 s_clim = s_clim[(s_clim.index >= t0) & (s_clim.index <= t1)]
#                 if not s_clim.empty:
#                     ln, = ax.plot(s_clim.index, s_clim.values, label='Climatology Data', alpha=0.7)
#                     has_data = True
#                     legend_handles.append(ln)

#             ax.text(0.02, 0.85, river_name, transform=ax.transAxes, ha='left', va='top',
#                     fontsize=12, fontweight='bold')

#             if not has_data:
#                 ax.axis('off')

#             plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
#             all_legend_handles.extend(legend_handles)

#         fig.subplots_adjust(hspace=1.2)
#         fig.text(0.04, 0.5, 'Discharge (m³/s)', va='center', rotation='vertical', fontsize=12)
#         fig.text(0.5, 0.04, 'Date', ha='center', fontsize=12)

#         if all_legend_handles:
#             fig.legend(handles=all_legend_handles[:2], labels=['River Data', 'Climatology Data'],
#                        loc='upper center', ncol=2, fontsize=12, frameon=False, bbox_to_anchor=(0.5, 1.02))

#         fig.savefig(save_path / f"climatology_vs_river_discharge_{fig_idx + 1}of{total_figs}.png",
#                     dpi=300, bbox_inches='tight')
#         plt.close(fig)


# # Example usage
# if __name__ == '__main__':
#     from pathlib import Path
#     from o_func import opsys
#     start_path = Path(opsys('PN'))
#     savepath = start_path / Path('modelling_DATA/kent_estuary_project/river_boundary_conditions/figures/CSVs')
#     plot_climatology(layout="portrait", num_figures=1, save_path=savepath, sharey=False)

def plot_climatology(layout="landscape", num_figures=1, save_path=".", sharey=False,
                     start_date="2013-10-28 00:00:00", end_date="2014-03-01 00:00:00",
                     rivers_per_figure=None, write_cumulative_csv=True, write_all_combos=True,
                     volume_units="auto", volume_decimal_places=None, unitless_decimal_places=2,
                     include_relative=False):
    """
    Writes cumulative discharge CSVs with fixed-decimal formatting. Columns:
      - river
      - cumulative_climatology_[UNIT]
      - cumulative_real_[UNIT]
      - cumulative_difference_[UNIT]   (climatology - real)
      - ratio_clim_over_real           (unitless, e.g. 3.50 means 3.5×)
      - [optional] relative_difference_to_real = (clim - real)/real

    Set include_relative=True to add the last column; default False (omitted).
    """
    import re
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from itertools import combinations
    from o_func import opsys

    # -------- helpers --------
    def norm(name: str) -> str:
        return re.sub(r'[^a-z0-9]+', '', str(name).lower())

    def canonical(key: str) -> str:
        key = norm(key)
        return {'clywd': 'clwyd'}.get(key, key)

    try:
        from numpy import trapezoid as _trapz
    except Exception:
        from numpy import trapz as _trapz

    def integrate_series(s: pd.Series) -> float:
        if s is None or s.empty:
            return np.nan
        s = s.dropna().sort_index()
        if s.empty:
            return np.nan
        idx = s.index
        if getattr(idx, 'tz', None) is not None:
            idx = idx.tz_convert('UTC').tz_localize(None)
        t = idx.astype('int64') / 1e9
        return float(_trapz(s.values, t))  # m^3

    def read_real_series(file_paths, t0, t1) -> pd.Series:
        frames = []
        for p in file_paths or []:
            df = pd.read_csv(p, header=None, names=["time", "value"])
            df['time'] = pd.to_datetime(df['time'])
            df = df[(df['time'] >= t0) & (df['time'] <= t1)]
            if not df.empty:
                frames.append(df[['time', 'value']].set_index('time').sort_index())
        if not frames:
            return pd.Series(dtype=float)
        merged = pd.concat(frames).groupby(level=0).mean().sort_index()
        return merged['value']

    def safe_ratio(clim, real, eps=1e-9):
        if pd.notna(clim) and pd.notna(real) and abs(real) > eps:
            return clim / real
        return np.nan

    def safe_rel_diff(clim, real, eps=1e-9):
        if pd.notna(clim) and pd.notna(real) and abs(real) > eps:
            return (clim - real) / real
        return np.nan

    def unit_scale_choice(max_val_m3: float, choice: str):
        table = {
            "m3":  ("m3",  "m^3",            1.0),
            "ML":  ("ML",  "ML (1e3 m^3)",   1e3),
            "GL":  ("GL",  "GL (1e6 m^3)",   1e6),
            "km3": ("km3", "km^3 (1e9 m^3)", 1e9),
        }
        if choice != "auto":
            key = choice
            return (key, table[key][1], table[key][2])
        if max_val_m3 >= 1e9:   key = "km3"
        elif max_val_m3 >= 1e6: key = "GL"
        elif max_val_m3 >= 1e3: key = "ML"
        else:                   key = "m3"
        return (key, table[key][1], table[key][2])

    def default_dp_for_unit(unit_key: str) -> int:
        return {"m3": 0, "ML": 1, "GL": 2, "km3": 3}[unit_key]

    def fmt_fixed(x, dp):
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return ""
        thresh = 0.5 * 10**(-dp)
        if abs(x) < thresh:
            x = 0.0
        return f"{x:.{dp}f}"

    # -------- paths & inputs --------
    start_path = Path(opsys('PN'))
    climatology_file_path = start_path / 'GitHub/o_func_install/o_func/data_prepkit/all_rivers.csv'
    rt = 'modelling_DATA/kent_estuary_project/river_boundary_conditions/delft_3dfm_inputs/001_river_data_2013-10-28_to_2014-03-01'
    other_riv = 'modelling_DATA/kent_estuary_project/river_boundary_conditions/delft_bc_files/extra_rivers_real_data_esk_alt_clywd/processed'
    river_path = start_path / rt
    other_river_path = start_path / other_riv

    climatology_df = pd.read_csv(climatology_file_path, index_col=0, parse_dates=True)
    clim_map = {canonical(c): climatology_df[c] for c in climatology_df.columns}
    display_from_clim = {canonical(c): c for c in climatology_df.columns}

    riv_files = {}
    display_from_files = {}
    for p in list(river_path.glob('*')) + list(other_river_path.glob('*')):
        if not p.is_file():
            continue
        base = p.name.split('_')[0]
        key = canonical(base)
        riv_files.setdefault(key, []).append(p)
        display_from_files.setdefault(key, base.title())

    all_keys = sorted(set(clim_map.keys()) | set(riv_files.keys()))
    combined = {}
    for key in all_keys:
        disp = display_from_clim.get(key, display_from_files.get(key, key.title()))
        combined[disp] = {
            "key": key,
            "river_data": riv_files.get(key, []),
            "climatology_data": clim_map.get(key, None),
        }

    t0 = pd.Timestamp(start_date)
    t1 = pd.Timestamp(end_date)

    if layout == "landscape":
        base_width, base_height = 14, 200
    elif layout == "portrait":
        base_width, base_height = 10, 250
    else:
        raise ValueError("Invalid layout")

    num_rivers = len(combined)
    if rivers_per_figure is None:
        rivers_per_figure = max(1, int(np.ceil(num_rivers / max(1, num_figures))))

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # -------- cumulative volumes --------
    rows = []
    for disp, data in combined.items():
        s_real = read_real_series(data["river_data"], t0, t1)
        s_clim = None
        if data["climatology_data"] is not None:
            s = data["climatology_data"]
            s_clim = s[(s.index >= t0) & (s.index <= t1)]
        v_real = integrate_series(s_real)  # m^3
        v_clim = integrate_series(s_clim)  # m^3
        v_diff = (v_clim - v_real) if (pd.notna(v_clim) or pd.notna(v_real)) else np.nan
        ratio = safe_ratio(v_clim, v_real)
        rel   = safe_rel_diff(v_clim, v_real)
        rows.append({
            "river": disp, "key": data["key"],
            "vol_clim_m3": v_clim, "vol_real_m3": v_real, "vol_diff_m3": v_diff,
            "ratio_clim_over_real": ratio, "relative_difference_to_real": rel
        })
    cum_df_all = pd.DataFrame(rows)

    def write_summary(label: str, exclude_keys=None):
        exclude_keys = set(canonical(x) for x in (exclude_keys or []))
        df = cum_df_all[~cum_df_all["key"].isin(exclude_keys)].copy()

        max_val = np.nanmax([
            df["vol_clim_m3"].abs().max(skipna=True),
            df["vol_real_m3"].abs().max(skipna=True),
            df["vol_diff_m3"].abs().max(skipna=True)
        ])
        if not np.isfinite(max_val): max_val = 0.0
        unit_key, unit_label, factor = unit_scale_choice(max_val, volume_units)
        vol_dp = default_dp_for_unit(unit_key) if volume_decimal_places is None else int(volume_decimal_places)
        rat_dp = int(unitless_decimal_places)

        # scale
        df[f"cumulative_climatology_[{unit_label}]"] = df["vol_clim_m3"] / factor
        df[f"cumulative_real_[{unit_label}]"]        = df["vol_real_m3"] / factor
        df[f"cumulative_difference_[{unit_label}]"]  = df["vol_diff_m3"] / factor

        # totals (scaled)
        total_clim = df[f"cumulative_climatology_[{unit_label}]"].sum(skipna=True)
        total_real = df[f"cumulative_real_[{unit_label}]"].sum(skipna=True)
        total_diff = total_clim - total_real
        total_ratio = safe_ratio(total_clim, total_real)
        total_rel   = safe_rel_diff(total_clim, total_real)

        # assemble output columns
        out_cols = ["river",
                    f"cumulative_climatology_[{unit_label}]",
                    f"cumulative_real_[{unit_label}]",
                    f"cumulative_difference_[{unit_label}]",
                    "ratio_clim_over_real"]
        if include_relative:
            out_cols.append("relative_difference_to_real")

        out = df[out_cols].copy()

        # append TOTAL (floats for now)
        total_row = {
            "river": "TOTAL",
            f"cumulative_climatology_[{unit_label}]": total_clim,
            f"cumulative_real_[{unit_label}]":        total_real,
            f"cumulative_difference_[{unit_label}]":  total_diff,
            "ratio_clim_over_real":                   total_ratio,
        }
        if include_relative:
            total_row["relative_difference_to_real"] = total_rel

        out = pd.concat([out, pd.DataFrame([total_row])], ignore_index=True)

        # fixed-decimal formatting
        vol_cols = [f"cumulative_climatology_[{unit_label}]",
                    f"cumulative_real_[{unit_label}]",
                    f"cumulative_difference_[{unit_label}]"]
        for c in vol_cols:
            out[c] = out[c].apply(lambda x: fmt_fixed(x, vol_dp))
        out["ratio_clim_over_real"] = out["ratio_clim_over_real"].apply(lambda x: fmt_fixed(x, rat_dp))
        if include_relative:
            out["relative_difference_to_real"] = out["relative_difference_to_real"].apply(lambda x: fmt_fixed(x, rat_dp))

        out_path = save_path / f"{label}.csv"
        out.to_csv(out_path, index=False)
        print(f"Wrote {out_path.name} using [{unit_label}] with {vol_dp} dp (volumes) / {rat_dp} dp (unitless).")
        return out_path

    if write_cumulative_csv:
        write_summary("all_rivers", exclude_keys=[])
        write_summary("no_duddon", exclude_keys=["duddon"])
        write_summary("no_clywd_esk_alt", exclude_keys=["clwyd", "clywd", "esk", "alt"])
        if write_all_combos:
            base_exclude_set = ["duddon", "clwyd", "esk", "alt"]
            seen = set()
            for r in range(1, len(base_exclude_set)+1):
                for combo in combinations(base_exclude_set, r):
                    canon_combo = tuple(sorted(canonical(x) for x in combo))
                    if canon_combo in seen: continue
                    seen.add(canon_combo)
                    write_summary("exclude_" + "_".join(canon_combo), exclude_keys=list(canon_combo))

    # -------- plotting (unchanged) --------
    items = list(combined.items())
    rivers_per_figure = max(1, rivers_per_figure) if rivers_per_figure is not None else max(1, int(np.ceil(num_rivers / max(1, num_figures))))
    total_figs = int(np.ceil(num_rivers / rivers_per_figure))
    for fig_idx in range(total_figs):
        start_idx = fig_idx * rivers_per_figure
        end_idx = min((fig_idx + 1) * rivers_per_figure, num_rivers)
        rivers_in_figure = items[start_idx:end_idx]
        num_rows = len(rivers_in_figure)
        fig, axes = plt.subplots(nrows=num_rows, ncols=1, figsize=(base_width, base_height),
                                 sharex=True, sharey=sharey)
        axes = axes.flatten() if num_rows > 1 else [axes]
        all_legend_handles = []
        for ax, (river_name, data) in zip(axes, rivers_in_figure):
            has_data = False; legend_handles = []
            if data['river_data']:
                s_real = read_real_series(data['river_data'], t0, t1)
                if not s_real.empty:
                    ln, = ax.plot(s_real.index, s_real.values, label='River Data', alpha=0.7)
                    has_data = True; legend_handles.append(ln)
            s_clim = data['climatology_data']
            if s_clim is not None:
                s_clim = s_clim[(s_clim.index >= t0) & (s_clim.index <= t1)]
                if not s_clim.empty:
                    ln, = ax.plot(s_clim.index, s_clim.values, label='Climatology Data', alpha=0.7)
                    has_data = True; legend_handles.append(ln)
            ax.text(0.02, 0.85, river_name, transform=ax.transAxes, ha='left', va='top', fontsize=12, fontweight='bold')
            if not has_data: ax.axis('off')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
            all_legend_handles.extend(legend_handles)
        fig.subplots_adjust(hspace=1.2)
        fig.text(0.04, 0.5, 'Discharge (m³/s)', va='center', rotation='vertical', fontsize=12)
        fig.text(0.5, 0.04, 'Date', ha='center', fontsize=12)
        if all_legend_handles:
            fig.legend(handles=all_legend_handles[:2], labels=['River Data', 'Climatology Data'],
                       loc='upper center', ncol=2, fontsize=12, frameon=False, bbox_to_anchor=(0.5, 1.02))
        fig.savefig(save_path / f"climatology_vs_river_discharge_{fig_idx + 1}of{total_figs}.png",
                    dpi=300, bbox_inches='tight')
        plt.close(fig)


# Example
if __name__ == '__main__':
    from pathlib import Path
    from o_func import opsys
    start_path = Path(opsys('PN'))
    savepath = start_path / Path('modelling_DATA/kent_estuary_project/river_boundary_conditions/figures/CSVs')
    # exclude relative column by default:
    plot_climatology(layout="portrait", num_figures=1, save_path=savepath, sharey=False,
                     volume_units="km3", include_relative=False, unitless_decimal_places= 3)
