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

    # Add labels
    axes[-1].set_xlabel('Date', fontsize=14, labelpad=15)  # Increased font size and added padding
    fig.text(0.04, 0.5, 'Discharge (m³/s)', va='center', rotation='vertical', fontsize=14)  # Increased font size and padding

    # Rotate x-axis labels diagonally
    for ax in axes:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add legend closer to the figure
    fig.legend(['River Data', 'Climatology Data'], loc='upper center', ncol=2, fontsize=12, bbox_to_anchor=(0.5, 0.91))

    # Save the figure
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    figure_path = save_path / "climatology_vs_river_discharge.png"
    fig.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Figure saved to {figure_path}")


if __name__ == '__main__':
    # Specify save path
    start_path = Path(opsys('PN'))
    savepath = start_path / Path('modelling_DATA/kent_estuary_project/river_boundary_conditions/figures')
    plot_climatology_simple(save_path=savepath)
