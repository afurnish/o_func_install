import xarray as xr
import os
import pandas as pd

def split_nc_by_day(input_file, output_dir=None):
    # Open the source NetCDF file using xarray
    ds = xr.open_dataset(input_file)
    ds = ds.swap_dims({'time': 'time_primea'}).set_index(time_primea='time_primea')
    # Filter for variables that start with 'prim_'
    ds_filtered = ds[[var for var in ds.variables if var.startswith('prim_')]]
    
    # If no output directory is provided, create a default one
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_file), "daily_netcdfs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by day and iterate through each day
    grouped = ds_filtered.groupby('time_primea.date')
    for date, daily_ds in grouped:
        # Format the date as YYYYMMDD
        date_str = pd.to_datetime(str(date)).strftime('%Y%m%d')
        
        # Define the output filename
        output_file = os.path.join(output_dir, f"{os.path.basename(input_file)[:-3]}_{date_str}_{date_str}_extended_grid.nc")
        
        # Save the dataset for that day
        daily_ds.to_netcdf(output_file)
        print(f"Created {output_file}")
    
    # Close the dataset
    ds.close()

# Usage example

split_nc_by_day(r'/media/af/Elements/INVEST_modelling/modelling_data/1.3d_models/Thoms_M2_models/M2_10m_3layers/kent_regrid.nc')
