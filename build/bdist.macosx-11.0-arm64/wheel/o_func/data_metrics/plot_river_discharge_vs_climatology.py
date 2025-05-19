def plot_climatology(layout="landscape", num_figures=1, save_path=".", sharey=False):
    import os
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt
    from datetime import datetime
    import numpy as np
    from o_func import opsys

    # Define paths using opsys
    start_path = Path(opsys('PN'))
    climatology_file_path = start_path / 'GitHub/o_func_install/o_func/data_prepkit/all_rivers.csv'
    rt = 'modelling_DATA/kent_estuary_project/river_boundary_conditions/delft_3dfm_inputs/001_river_data_2013-10-28_to_2014-03-01'
    other_riv = 'modelling_DATA/kent_estuary_project/river_boundary_conditions/delft_bc_files/extra_rivers_real_data_esk_alt_clywd/processed'
    river_path = start_path / Path(rt)
    other_river_path = start_path / Path(other_riv)

    # Load climatology data from the provided CSV file
    climatology_df = pd.read_csv(climatology_file_path, index_col=0, parse_dates=True)

    # Initialize dictionaries to store data
    riv_files = {}
    combined_data = {}

    # Collect and organize file paths for river data
    for file_riv in river_path.glob('*'):
        river_name = file_riv.name.split('_')[0].lower()
        riv_files.setdefault(river_name, []).append(file_riv)

    # Handle Alt, Clywd, Esk
    for file_riv in other_river_path.glob('*'):
        river_name = file_riv.name.split('_')[0].lower()
        riv_files.setdefault(river_name, []).append(file_riv)

    # Combine into a single dictionary, using climatology data from the CSV
    all_river_names = sorted(riv_files.keys())  # Sort river names alphabetically
    for river_name in all_river_names:
        combined_data[river_name] = {
            'river_data': riv_files.get(river_name, []),
            'climatology_data': climatology_df[river_name.capitalize()] if river_name.capitalize() in climatology_df.columns else None
        }

    # Define start and end dates
    start_date = pd.Timestamp("2013-10-28 00:00:00")
    end_date = pd.Timestamp("2014-03-01 00:00:00")

    # Adjust layout settings based on chosen orientation
    if layout == "landscape":
        base_width, base_height = 14, 200  # Make the figure significantly taller
    elif layout == "portrait":
        base_width, base_height = 10, 250  # Make the figure significantly taller
    else:
        raise ValueError("Invalid layout choice. Use 'landscape' or 'portrait'.")

    num_rivers = len(combined_data)
    rivers_per_figure = (num_rivers + num_figures - 1) // num_figures

    # Ensure the save path exists
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Create multiple figures as needed
    for fig_idx in range(num_figures):
        start_idx = fig_idx * rivers_per_figure
        end_idx = min((fig_idx + 1) * rivers_per_figure, num_rivers)
        rivers_in_figure = list(combined_data.items())[start_idx:end_idx]

        num_rows = len(rivers_in_figure)  # One subplot per row

        # Create the figure
        fig, axes = plt.subplots(nrows=num_rows, ncols=1, figsize=(base_width, base_height), sharex=True, sharey=sharey)
        axes = axes.flatten() if num_rows > 1 else [axes]  # Ensure axes is iterable

        all_legend_handles = []

        for idx, (river_name, data) in enumerate(rivers_in_figure):
            ax = axes[idx]
            has_data = False
            legend_handles = []

            # Plot river data if available
            if data['river_data']:
                for river_file in data['river_data']:
                    river_data = pd.read_csv(river_file, header=None, names=["time", "value"])
                    river_data['time'] = pd.to_datetime(river_data['time'])
                    river_data = river_data[(river_data['time'] >= start_date) & (river_data['time'] <= end_date)]
                    if not river_data.empty:
                        line, = ax.plot(river_data['time'], river_data['value'], label='River Data', color='blue', alpha=0.7)
                        has_data = True
                        legend_handles.append(line)

            # Plot climatology data if available
            if data['climatology_data'] is not None:
                climatology_series = data['climatology_data'][
                    (climatology_df.index >= start_date) & (climatology_df.index <= end_date)]
                if not climatology_series.empty:
                    climatology_values = climatology_series.values
                    line, = ax.plot(climatology_series.index, climatology_values, label='Climatology Data', color='red', alpha=0.7)
                    has_data = True
                    legend_handles.append(line)

            # Annotate the estuary name
            ax.text(0.02, 0.85, river_name.capitalize(), transform=ax.transAxes, ha='left', va='top',
                    fontsize=12, fontweight='bold')

            # Collect legend handles for global legend
            all_legend_handles.extend(legend_handles)

            # Hide the subplot if no data is available
            if not has_data:
                ax.axis('off')

            # Rotate x-axis dates diagonally
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Adjust subplot spacing
        fig.subplots_adjust(hspace=1.2)  # Increase spacing between rows

        # Add a single y-axis label for the figure
        fig.text(0.04, 0.5, 'Discharge (m³/s)', va='center', rotation='vertical', fontsize=12)

        # Set common x-label
        fig.text(0.5, 0.04, 'Date', ha='center', fontsize=12)

        # Add a single legend for the entire figure
        if all_legend_handles:
            fig.legend(handles=all_legend_handles[:2], labels=['River Data', 'Climatology Data'],
                       loc='upper center', ncol=2, fontsize=12, frameon=False, bbox_to_anchor=(0.5, 1.02))

        # Save the figure to the specified folder
        figure_name = f"climatology_vs_river_discharge_{fig_idx + 1}.png"
        fig.savefig(save_path / figure_name, dpi=300, bbox_inches='tight')  # Force saving with exact dimensions
        plt.close(fig)  # Close the figure after saving to avoid display issues


if __name__ == '__main__':
    from pathlib import Path 
    from o_func import opsys
    # Choose layout: 'landscape' or 'portrait', number of figures, and save path
    start_path = Path(opsys('PN'))
    savepath = start_path / Path('modelling_DATA/kent_estuary_project/river_boundary_conditions/figures')
    plot_climatology(layout="portrait", num_figures=1, save_path=savepath, sharey=False)
