#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Regridding program used to translate delft 3dfm suite model outputs using UKC4 rectalinear grids.
Created on Mon Jan 29 11:51:32 2024
@author: af

To handle the bathymetry component I am quite happy with this formula
Bathymetry=−(Water Depth−Surface Height)
"""

# import xarray as xr
# import os
# os.chdir('D:/INVEST_modelling/modelling_data/regrid_3d_testing')
# data = xr.open_dataset('kent_31_subsampled_map.nc')


#%% Import Dependencies
import time
import numpy as np
import sys
import os
import glob
import pickle
from multiprocessing import Pool

from tqdm import tqdm

from scipy.interpolate import CloughTocher2DInterpolator
import xarray as xr
import pyproj

#%% Functions
def line():
    print('-'*60)

start = time.time()

#%% Run locally or on SCW

var_dict = {
    'surface_height': {'TUV': 'T', 'UKC4': 'sossheig', 'PRIMEA': 'mesh2d_s1', 'UNITS': 'm', 'layers': False},
    'surface_salinity': {'TUV': 'T', 'UKC4': 'vosaline_top', 'PRIMEA': 'mesh2d_sa1', 'UNITS': 'psu', 'layers': True},
    'middle_salinity': {'TUV': 'T', 'UKC4': 'vosaline_mid', 'PRIMEA': 'na', 'UNITS': 'psu', 'layers': True},
    'bottom_salinity': {'TUV': 'T', 'UKC4': 'vosaline_bot', 'PRIMEA': 'na', 'UNITS': 'psu', 'layers': True},
    'surface_temp': {'TUV': 'T', 'UKC4': 'votemper_top', 'PRIMEA': 'na', 'UNITS': '°C', 'layers': True},
    'middle_temp': {'TUV': 'T', 'UKC4': 'votemper_mid', 'PRIMEA': 'na', 'UNITS': '°C', 'layers': True},
    'bottom_temp': {'TUV': 'T', 'UKC4': 'votemper_bot', 'PRIMEA': 'na', 'UNITS': '°C', 'layers': True},
    'surface_Uvelocity': {'TUV': 'U', 'UKC4': 'vozocrtx_top', 'PRIMEA': 'mesh2d_ucx', 'UNITS': 'm', 'layers': True},
    'middle_Uvelocity': {'TUV': 'U', 'UKC4': 'vozocrtx_mid', 'PRIMEA': 'na', 'UNITS': 'm', 'layers': True},
    'bottom_Uvelocity': {'TUV': 'U', 'UKC4': 'vozocrtx_bot', 'PRIMEA': 'na', 'UNITS': 'm', 'layers': True},
    'surface_Vvelocity': {'TUV': 'V', 'UKC4': 'vomecrty_top', 'PRIMEA': 'mesh2d_ucy', 'UNITS': 'm', 'layers': True},
    'middle_Vvelocity': {'TUV': 'V', 'UKC4': 'vomecrty_mid', 'PRIMEA': 'na', 'UNITS': 'm', 'layers': True},
    'bottom_Vvelocity': {'TUV': 'V', 'UKC4': 'vomecrty_bot', 'PRIMEA': 'na', 'UNITS': 'm', 'layers': True},
    'bathymetry': {'TUV': 'T', 'UKC4': 'NA', 'PRIMEA': 'mesh2d_node_z', 'UNITS': 'm', 'layers': False},
}

# Define input and output paths
input_file = 'kent_31_merged_map.nc'
output_nc_file = 'kent_regrid.nc'

print('Loading PRIMEA datasets...')
primea_model = xr.open_dataset(input_file)

# Load UKC4 datasets
mp = os.path.join('/','home',r'b.osu903','kent','oa','shelftmb_cut_to_domain')
T = glob.glob(os.path.join(mp, "*T.nc"))
U = glob.glob(os.path.join(mp, "*U.nc"))
V = glob.glob(os.path.join(mp, "*V.nc"))
print('Loading UKC4 datasets...')
TUV = [xr.open_mfdataset(i) for i in [T,U,V]]

lonTUV = [i.nav_lon for i in TUV]
latTUV = [i.nav_lat for i in TUV]

plon = primea_model.mesh2d_s1.mesh2d_face_x
plat = primea_model.mesh2d_s1.mesh2d_face_y

bathyplon = primea_model.mesh2d_node_z.mesh2d_node_x
bathyplat = primea_model.mesh2d_node_z.mesh2d_node_y

projector = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
x_unstructured, y_unstructured = projector.transform(np.array(plon).flatten(), np.array(plat).flatten())
x_bathy_unstructured, y_bathy_unstructured = x_unstructured, y_unstructured

x_structuredT, y_structuredT = projector.transform(np.array(lonTUV[0]).flatten(), np.array(latTUV[0]).flatten())
x_structuredU, y_structuredU = projector.transform(np.array(lonTUV[1]).flatten(), np.array(latTUV[1]).flatten())
x_structuredV, y_structuredV = projector.transform(np.array(lonTUV[2]).flatten(), np.array(latTUV[2]).flatten())

line()
first_iteration = -1
print('Running regridding operation...')
interpolator_pkl = os.path.join(os.path.split(input_file)[0],'interpolator.pkl')

def para_proc(lm):
    names = []
    primea_saves = []
    for jk, k in enumerate([i for i in var_dict]):
        # perform a check to ensure the variables exist
        if var_dict[k]['PRIMEA'] in [i for i in primea_model.data_vars]:
            var = var_dict[k]['PRIMEA']
            if lm == 0:
                names.append(k) # keeps order of variables comparing.
            else:
                names.append('n/a')
            
            x_structured, y_structured = None, None
            lon = None
            if var_dict[k]['TUV'] == 'T':
                x_structured, y_structured = x_structuredT, y_structuredT
                lon = lonTUV[0]
            elif var_dict[k]['TUV'] == 'U':
                x_structured, y_structured = x_structuredU, y_structuredU
                lon = lonTUV[1]
            elif var_dict[k]['TUV'] == 'V':
                x_structured, y_structured = x_structuredV, y_structuredV
                lon = lonTUV[2]

            interpolated_values_reshaped = []
            if k != 'bathymetry':
                if var_dict[k]['layers']:
                    for layer in range(primea_model[var].shape[2]):
                        interpolator = CloughTocher2DInterpolator((x_unstructured, y_unstructured), np.array(primea_model[var][lm, :, layer]).flatten())
                        values = interpolator(x_structured, y_structured)
                        interpolated_values_reshaped.append(values.reshape(lon.shape))
                else:
                    interpolator = CloughTocher2DInterpolator((x_unstructured, y_unstructured), np.array(primea_model[var][lm]).flatten())
                    values = interpolator(x_structured, y_structured)
                    interpolated_values_reshaped.append(values.reshape(lon.shape))
            else: # Bathymetry
                new_bathy_proxy = primea_model['mesh2d_waterdepth'][lm] - primea_model['mesh2d_s1'][lm]
                interpolator = CloughTocher2DInterpolator((x_bathy_unstructured, y_bathy_unstructured), np.array(new_bathy_proxy).flatten())
                values = interpolator(x_structured, y_structured)
                interpolated_values_reshaped.append(values.reshape(lon.shape))
            
            primea_saves.append(np.stack(interpolated_values_reshaped, axis=-1))
    return lm, primea_saves, names

num_iterations = len(primea_model.time)

#%% Parallelisation
primea_time = primea_model.time[:num_iterations]

try:
    primea_time_testing = primea_time[2:-2]
    disorder_indices = np.where(np.diff(primea_time_testing) < np.timedelta64(0))[0]
    if len(disorder_indices) > 0:
        raise ValueError(f"Error: primea_time is not in chronological order at indices {disorder_indices}.")

    time_diffs = np.diff(primea_time_testing)
    inconsistent_intervals_indices = np.where(time_diffs != time_diffs[0])[0]
    if len(inconsistent_intervals_indices) > 0:
        raise ValueError(f"Error: Intervals between time steps are not constant at indices {inconsistent_intervals_indices}.")

except ValueError as e:
    print(e)
    print('Time is not in chronological order, model may be broken or tampered with...')

with Pool() as pool:
    packed_results = list(tqdm(pool.imap_unordered(para_proc, range(num_iterations)), total=num_iterations))

pool.close()
pool.join()

results_sorted = sorted(packed_results, key=lambda x: x[0])
indices, results, names = zip(*results_sorted)
seperated_results = list(zip(*results))
filtered_names = [row for row in names if 'n/a' not in row][0]

empty_PRIMEA_data_array = []
empty_UKC4_data_array = []

for kl, nme in enumerate(tqdm(filtered_names, desc="Processing UKC4 data: ", unit="file")):
    index_of_bathymetry = filtered_names.index('bathymetry')

    if var_dict[nme]['TUV'] == 'T':
        if kl != index_of_bathymetry:
            values, rows, cols = rcr_PRIMEA(seperated_results[kl])
        else:
            values, rows, cols = rcr_PRIMEA(seperated_results[kl], import_rowcol = 'y', r=rows, c = cols)
        if nme != 'bathymetry':
            ukc4_dataset = rcr_UKC4(TUV[0][var_dict[nme]['UKC4']], rows, cols)
            empty_UKC4_data_array.append(ukc4_dataset)
            vals = mask_maker(ukc4_dataset, values)
        if nme == 'bathymetry':
            vals = mask_maker(ukc4_dataset, values)
            empty_PRIMEA_data_array.append(np.stack(vals, axis=0))
        else:
            empty_PRIMEA_data_array.append(np.stack(vals, axis=0))

    if var_dict[nme]['TUV'] == 'U':
        vals, rows, cols = rcr_PRIMEA(seperated_results[kl])
        ukc4_dataset = rcr_UKC4(TUV[1][var_dict[nme]['UKC4']], rows, cols)
        empty_UKC4_data_array.append(ukc4_dataset)
        vals = mask_maker(ukc4_dataset, vals)
        empty_PRIMEA_data_array.append(np.stack(vals, axis=0))

    if var_dict[nme]['TUV'] == 'V':
        vals, rows, cols = rcr_PRIMEA(seperated_results[kl])
        ukc4_dataset = rcr_UKC4(TUV[2][var_dict[nme]['UKC4']], rows, cols)
        empty_UKC4_data_array.append(ukc4_dataset)
        vals = mask_maker(ukc4_dataset, vals)
        empty_PRIMEA_data_array.append(np.stack(vals, axis=0))

print('ukc4 size', len(empty_UKC4_data_array))
print('prim size', len(empty_PRIMEA_data_array))

ukc4_xrdata_array = [xr.concat(i, dim='time_counter') for i in empty_UKC4_data_array]
primea_data_array = empty_PRIMEA_data_array

time_coord = xr.DataArray(primea_time, dims='time', coords={'time': primea_time})

latlon_2d = [xr.broadcast(ukc4_xrdata_array[i]['nav_lon'], ukc4_xrdata_array[i]['nav_lat']) for i in range(len(ukc4_xrdata_array))]
grid_mapping = {}
for lon, lat in latlon_2d:
    grid_type = lat.nav_model.split('_')[-1]
    grid_mapping[grid_type] = {'lon': lon, 'lat': lat}
variable_grid_types = [var_dict[name]['TUV'] for name in filtered_names]

primea_converted_data_array = [xr.DataArray(
                        i,
                        dims=('time', 'y', 'x', 'layer' if var_dict[filtered_names[idx]]['layers'] else None),
                        coords={
                            'nav_lat': (('y', 'x'), grid_mapping[variable_grid_types[idx]]['lat'].data),
                            'nav_lon': (('y', 'x'), grid_mapping[variable_grid_types[idx]]['lon'].data),
                            'time_primea': time_coord
                            },
                                    ) for idx, i in enumerate(primea_data_array)]

primea_attrs_list = [primea_model[var_dict[i]['PRIMEA']] for i in filtered_names]
for i, k in enumerate(primea_converted_data_array):
    k.attrs = primea_attrs_list[i].attrs

dict_to_dataset = {}
for ik, dataset in enumerate([primea_converted_data_array, ukc4_xrdata_array]):
    for ig, name in enumerate(filtered_names):
        if ik == 0:
            key = 'prim_' + name
            print(key)
        else:
            if name != 'bathymetry':
                key = 'ukc4_' + name
        if ig < len(dataset):
            dict_to_dataset[key] = dataset[ig]

renamed_coord_dict_to_dataset = rename_coordinates_for_velocity_datasets(dict_to_dataset)

def save_regrid():
    dataset = xr.Dataset(renamed_coord_dict_to_dataset)

    if os.path.exists(output_nc_file):
        os.remove(output_nc_file)
        print(f"Old File '{output_nc_file}' deleted.")
    line()
    print('Writing to file...')
    dataset.to_netcdf(output_nc_file, mode='w')

    minutes, seconds = divmod((time.time() - start), 60)
    formatted_time = "{:02}:{:02}".format(int(minutes), int(seconds))

    print('Finished regridding in', formatted_time)

if __name__ == '__main__':
    save_regrid()
