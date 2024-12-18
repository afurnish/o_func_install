def old_climatology():
    import os
    from o_func import opsys
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt
    from datetime import datetime
    import numpy as np
    
    # Define paths using opsys
    start_path = Path(opsys('PN'))
    ct = 'modelling_DATA/kent_estuary_project/river_boundary_conditions/delft_bc_files/amm15_river_climatology'
    rt = 'modelling_DATA/kent_estuary_project/river_boundary_conditions/delft_3dfm_inputs/001_river_data_2013-10-28_to_2014-03-01'
    other_riv = 'modelling_DATA/kent_estuary_project/river_boundary_conditions/delft_bc_files/extra_rivers_real_data_esk_alt_clywd/processed'
    climatology_path = start_path / Path(ct)
    river_path = start_path / Path(rt)
    other_river_path = start_path / Path(other_riv)
    
    # Initialize dictionaries to store data
    riv_files = {}
    clim_files = {}
    combined_data = {}
    
    # Collect and organize file paths separately for climatology and river files
    for file_riv in river_path.glob('*'):
        river_name = file_riv.name.split('_')[0].lower()
        riv_files.setdefault(river_name, []).append(file_riv)
    
    # Handle Alt, Clywd, Esk
    for file_riv in other_river_path.glob('*'):
        river_name = file_riv.name.split('_')[0].lower()
        riv_files.setdefault(river_name, []).append(file_riv)
    
    for file_clim in climatology_path.glob('*'):
        climatology_name = file_clim.stem.lower()
        clim_files.setdefault(climatology_name, []).append(file_clim)
    
    # Combine into a single dictionary
    all_river_names = sorted(set(riv_files.keys()).union(clim_files.keys()))  # Sort river names alphabetically
    for river_name in all_river_names:
        combined_data[river_name] = {
            'river_data': riv_files.get(river_name, []),
            'climatology_data': clim_files.get(river_name, [])
        }
    
    # Function to convert climatology values to discharge units
    def convert_clim_to_discharge_units(array):
        length = 1500
        width = 1500
        density = 1000
        area = length * width
        array = np.array(array)  # Convert the list to a numpy array
        flow_rate = (array * area) / density
        return flow_rate
    
    # Define start and end dates
    start_date = pd.Timestamp("2013-10-28 00:00:00")
    end_date = pd.Timestamp("2014-03-01 00:00:00")
    
    # Create a multi-panel plot with two columns
    num_rivers = len(combined_data)
    num_columns = 2
    num_rows = (num_rivers + 1) // num_columns
    
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_columns, figsize=(18, num_rows * 4), sharex=True)
    axes = axes.flatten()  # Flatten to simplify indexing
    
    # Plotting each river on a separate subplot
    for idx, (river_name, data) in enumerate(combined_data.items()):
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
        if data['climatology_data']:
            for clim_file in data['climatology_data']:
                with open(clim_file, 'r') as file:
                    climatology_data = file.readlines()
                climatology_dates = []
                climatology_values = []
                for line in climatology_data:
                    date_str, value = line.strip().split(',')
                    try:
                        parsed_date = datetime.strptime(date_str, "%m-%d")
                        climatology_dates.append(parsed_date)
                        climatology_values.append(float(value))
                    except ValueError:
                        continue
    
                # Split the climatology data at June and adjust years
                climatology_values = convert_clim_to_discharge_units(climatology_values)
                jan_to_june_dates = [date.replace(year=2014) for date in climatology_dates if date.month <= 6]
                jan_to_june_values = [climatology_values[i] for i, date in enumerate(climatology_dates) if date.month <= 6]
                july_to_dec_dates = [date.replace(year=2013) for date in climatology_dates if date.month > 6]
                july_to_dec_values = [climatology_values[i] for i, date in enumerate(climatology_dates) if date.month > 6]
    
                # Combine adjusted dates and values
                adjusted_dates = july_to_dec_dates + jan_to_june_dates
                adjusted_values = july_to_dec_values + jan_to_june_values
    
                adjusted_df = pd.DataFrame({'date': adjusted_dates, 'value': adjusted_values})
                adjusted_df = adjusted_df[(adjusted_df['date'] >= start_date) & (adjusted_df['date'] <= end_date)]
                if not adjusted_df.empty:
                    line, = ax.plot(adjusted_df['date'], adjusted_df['value'], label='Climatology Data', color='red', alpha=0.7)
                    has_data = True
                    legend_handles.append(line)
    
        # Annotate the estuary name
        ax.text(0.02, 0.85, river_name.capitalize(), transform=ax.transAxes, ha='left', va='top',
                fontsize=12, fontweight='bold')
    
        # Add the legend to the top right of the subplot
        if legend_handles:
            ax.legend(handles=legend_handles, loc='upper right')
    
        # Hide the subplot if no data is available
        if not has_data:
            ax.axis('off')
    
        ax.set_ylabel('Discharge (m³/s)')
    
    # Hide the last empty subplot
    if len(axes) > num_rivers:
        axes[num_rivers].axis('off')
    
    # Set common x-axis limits for all subplots
    plt.xlim([start_date, end_date])
    
    # Set common x-label
    fig.text(0.5, 0.04, 'Date', ha='center')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust layout to make room for labels
    plt.show()

#%% 
def new_climatology():
    import os
    from o_func import opsys
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt
    from datetime import datetime
    import numpy as np
    
    # Define paths using opsys
    start_path = Path(opsys('PN'))
    rt = 'modelling_DATA/kent_estuary_project/river_boundary_conditions/delft_3dfm_inputs/001_river_data_2013-10-28_to_2014-03-01'
    other_riv = 'modelling_DATA/kent_estuary_project/river_boundary_conditions/delft_bc_files/extra_rivers_real_data_esk_alt_clywd/processed'
    river_path = start_path / Path(rt)
    other_river_path = start_path / Path(other_riv)
    
    # Load climatology data from the provided CSV file
    climatology_file_path = start_path / 'GitHub/o_func_install/o_func/data_prepkit/all_rivers.csv'
    climatology_df = pd.read_csv(climatology_file_path, index_col = 0,parse_dates=True)
    
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
    
    # Function to convert climatology values to discharge units
    # def convert_clim_to_discharge_units(array):
    #

if __name__ == '__main__':
    old_climatology()
    new_climatology() 