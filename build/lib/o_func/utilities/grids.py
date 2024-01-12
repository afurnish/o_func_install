import xarray as xr
import numpy as np
from scipy.interpolate import griddata, CloughTocher2DInterpolator
import pyproj
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
from tqdm import tqdm

#grid_example = '/Volumes/PN/modelling_DATA/kent_estuary_project/6.Final2/models/02_kent_1.0.0_UM_wind/shortrunSCW_kent_1.0.0_UM_wind/UK_West+Duddon+Raven_+liv+ribble+wyre+ll+ul+leven_kent_1.1_net.nc'

#data = xr.open_dataset(grid_example)

def is_counterclockwise(A, B, C):
    area = 0.5 * ((B - A) * (C - A) - (C - A) * (B - A))
    print(area)
    return area > 0

# # Assuming data is your xarray DataArray
# face_nodes = data['mesh2d_face_nodes'].values
# contains_nan = np.any(np.isnan(face_nodes), axis=1)

# # Separate triangles and squares
# triangles = face_nodes[contains_nan]
# squares = face_nodes[~contains_nan]

# for i in range(len(triangles)):
#     # Assuming x-y coordinates only
#     A, B, C = triangles[i, 0], triangles[i, 1], triangles[i, 2]

#     # Check counterclockwise condition
#     triangle_orientation = is_counterclockwise(A, B, C)
#     print(f'Triangle {i + 1}: Counterclockwise = {triangle_orientation}')

def PRIMEA_to_ESMF():
    """
    Created on Thu Oct  5 13:40:54 2023

    @author: af
    """
    # Import dependecies
    import netCDF4 as nc
    import numpy as np
    import xarray as xr
    import glob

    #Load in file of only the grid 
    file_grid = r'*merged*.nc'
    find_grid = glob.glob(file_grid)
    print(find_grid)
    data3 = xr.open_dataset (find_grid[0], engine = 'scipy')


    # last corner = 3, remap out grid points
    for numer, i in enumerate(data3.mesh2d_face_x_bnd[:,3]):
        if np.isnan(data3.mesh2d_face_x_bnd[numer,3]) == True:
            if numer % 1000 == 0:
                print(numer)
            val = data3.mesh2d_face_x_bnd[numer,0].item()
            data3.mesh2d_face_x_bnd[numer,3] = val
    for numer, i in enumerate(data3.mesh2d_face_y_bnd[:,3]):
        if np.isnan(data3.mesh2d_face_y_bnd[numer,3]) == True:
            val = data3.mesh2d_face_y_bnd[numer,0].item()
            data3.mesh2d_face_y_bnd[numer,3] = val
            


    # Apply grid points to vars that can then be readded back into the regridding file.
    grid_corner_lon = data3.mesh2d_face_x_bnd
    grid_corner_lat = data3.mesh2d_face_y_bnd

    data3['grid_centre_lon'] = data3.mesh2d_face_x
    data3['grid_centre_lat'] = data3.mesh2d_face_y
    data3['grid_corner_lon'] = data3.mesh2d_face_x_bnd
    data3['grid_corner_lat'] = data3.mesh2d_face_y_bnd


    data3.rename_dims({'max_nmesh2d_face_nodes':'grid_corners',
                      'nmesh2d_face':'grid_size'
                      })

    ### This is the last step but it needs to look like this. 
    data3.to_netcdf('testing_grid_corner.nc')

    # Replace data5 with the data from your array to create grid definition file. 

    # Create a new NetCDF file
    with nc.Dataset('PRIMEAgrid.nc', 'w', format='NETCDF4') as rootgrp:
        # Define dimensions
        rootgrp.createDimension('grid_rank', 2) # was 1
        rootgrp.createDimension('grid_size', data3.mesh2d_face_x.shape[0])
        rootgrp.createDimension('grid_corners', 4)
        #rootgrp.createDimension('grid_dim', 1) # was 1
        

        # Define variables
        grid_dims = rootgrp.createVariable('grid_dims', 'i4', ('grid_rank'))

        #grid_dims = rootgrp.createVariable('grid_dims', 'i4', ('grid_rank', 'grid_rank'))
        grid_center_lat = rootgrp.createVariable('grid_center_lat', 'f8', ('grid_size',))
        grid_center_lon = rootgrp.createVariable('grid_center_lon', 'f8', ('grid_size',))
        #rid_imask = rootgrp.createVariable('grid_imask', 'i4', ('grid_size',))
        grid_corner_lat = rootgrp.createVariable('grid_corner_lat', 'f8', ('grid_size', 'grid_corners'))
        grid_corner_lon = rootgrp.createVariable('grid_corner_lon', 'f8', ('grid_size', 'grid_corners'))

        # Add variable attributes
        grid_dims.long_name = 'grid_dims'
        grid_center_lat.long_name = 'grid_center_lat'
        grid_center_lat.units = 'degrees'
        grid_center_lon.long_name = 'grid_center_lon'
        grid_center_lon.units = 'degrees'
        #grid_imask.long_name = 'grid_imask'
        #grid_imask.units = 'unitless'
        grid_corner_lat.long_name = 'grid_corner_lat'
        grid_corner_lat.units = 'degrees'
        grid_corner_lon.long_name = 'grid_corner_lon'
        grid_corner_lon.units = 'degrees'

        # Add global attributes
        rootgrp.setncattr('NetCDF_source_t', 'PRIMEAgrid.nc')
        rootgrp.setncattr('file_name', 'PRIMEAgrid.nc')
        rootgrp.setncattr('title', 'PRIMEA')

        # You can now assign data to these variables using numpy arrays, for example:
        ##print(grid_dims)
        ##grid_dims[:] = [2] #  This was 2
        grid_center_lat[:] = data3.mesh2d_face_y.values
        grid_center_lon[:] = data3.mesh2d_face_x.values
        # grid_imask[:] = your_mask_data  # Replace with actual mask data
        grid_corner_lat[:] = data3.mesh2d_face_y_bnd  # Replace with actual corner latitude data
        grid_corner_lon[:] = data3.mesh2d_face_x_bnd  #


    ''' Temmplate to work off of 

    netcdf ocnUKO2_T_grid_out_nom {
    dimensions:
    	grid_rank = 2 ;
    	grid_size = 1961010 ;
    	grid_corners = 4 ;
    variables:
    	int grid_dims(grid_rank) ;
    		grid_dims:long_name = "grid_dims" ;
    	double grid_center_lat(grid_size) ;
    		grid_center_lat:long_name = "grid_center_lat" ;
    		grid_center_lat:units = "degrees" ;
    	double grid_center_lon(grid_size) ;
    		grid_center_lon:long_name = "grid_center_lon" ;
    		grid_center_lon:units = "degrees" ;
    	int grid_imask(grid_size) ;
    		grid_imask:long_name = "grid_imask" ;
    		grid_imask:units = "unitless" ;
    	double grid_corner_lat(grid_size, grid_corners) ;
    		grid_corner_lat:long_name = "grid_corner_lat" ;
    		grid_corner_lat:units = "degrees" ;
    	double grid_corner_lon(grid_size, grid_corners) ;
    		grid_corner_lon:long_name = "grid_corner_lon" ;
    		grid_corner_lon:units = "degrees" ;

    // global attributes:
    		:NetCDF_source_t = "ocnUKO2_T_grid_out_nom.nc" ;
    		:file_name = "ocnUKO2_T_grid_out_nom.nc" ;
    		:title = "AMM15T"
            
            
            
    PYTHON VIEW 
    data5
    Out[21]: 
    <xarray.Dataset>
    Dimensions:          (grid_rank: 2, grid_size: 1961010, grid_corners: 4)
    Dimensions without coordinates: grid_rank, grid_size, grid_corners
    Data variables:
        grid_dims        (grid_rank) int32 ...
        grid_center_lat  (grid_size) float64 ...
        grid_center_lon  (grid_size) float64 ...
        grid_imask       (grid_size) int32 ...
        grid_corner_lat  (grid_size, grid_corners) float64 ...
        grid_corner_lon  (grid_size, grid_corners) float64 ...
    Attributes:
        NetCDF_source_t:  ocnUKO2_V_grid_out_nom.nc
        file_name:        ocnUKO2_V_grid_out_nom.nc
        title:            AMM15V
    '''
#%%    

if __name__ == '__main__':
    # This will open various ukc3 files as they have slightly different grids for comparrison. 
    # we will start with U since it is to hand. 
    U = xr.open_dataset("/media/af/PN/Original_Data/UKC3/sliced/oa/shelftmb_cut_to_domain/UKC4ao_1h_20131030_20131030_shelftmb_grid_U.nc")
    #V = xr.open_dataset("/media/af/PN/Original_Data/UKC3/sliced/oa/shelftmb_cut_to_domain/UKC4ao_1h_20131030_20131030_shelftmb_grid_V.nc")
    lon, lat = U.nav_lon, U.nav_lat
    
    
    primea_model = xr.open_dataset('/media/af/PN/modelling_DATA/kent_estuary_project/6.Final2/models/kent_1.0.0_UM_wind/kent_31_merged_map.nc', engine = 'scipy')
    #regridded_primea = '/media/af/PN/modelling_DATA/kent_estuary_project/6.Final2/models/kent_1.0.0_UM_wind/regridded_kent_31_merged_map.nc'
    
    #start with water 
    sh = primea_model.mesh2d_s1
    plon = primea_model.mesh2d_s1.mesh2d_face_x
    plat = primea_model.mesh2d_s1.mesh2d_face_y
    #ppoints = np.column_stack((np.array(plon).flatten(), np.array(plat).flatten()))

    projector = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x_unstructured, y_unstructured = projector.transform(np.array(plon).flatten(), np.array(plat).flatten())
    x_structured, y_structured = projector.transform(np.array(lon).flatten(), np.array(lat).flatten())
    
    print('Running regridding operation...')
    def para_proc(i):
        # values = griddata((x_unstructured, y_unstructured), np.array(sh[420]).flatten(), (x_structured, y_structured), method='cubic')
        interpolator = CloughTocher2DInterpolator((x_unstructured, y_unstructured), np.array(sh[i]).flatten())
        values = interpolator(x_structured, y_structured)
        
        interpolated_values_reshaped = values.reshape(lon.shape)
        #plt.pcolor(lon, lat, interpolated_values_reshaped)
    
        # Find rows and columns that contain only NaN values
        rows_to_remove = np.all(np.isnan(interpolated_values_reshaped), axis=1)
        cols_to_remove = np.all(np.isnan(interpolated_values_reshaped), axis=0)
        
        # Slice off the rows and columns with only NaN values to trim the fat of the image
        interpolated_values_sliced = interpolated_values_reshaped[~rows_to_remove, :]
        interpolated_values_sliced = interpolated_values_sliced[:, ~cols_to_remove]
        lons_sliced = lon[~rows_to_remove, :]
        lons_sliced = lons_sliced[:, ~cols_to_remove]
        lats_sliced = lat[~rows_to_remove, :]
        lats_sliced = lats_sliced[:, ~cols_to_remove]
        
        masked = U.vobtcrtx[20][~rows_to_remove, :]
        masked = masked[:, ~cols_to_remove]
        nan_mask = np.isnan(masked)
        mask_grid = np.where(nan_mask, np.nan, interpolated_values_sliced)
        # if i == 0:
        #     result_array = np.empty((0, mask_grid.shape[0], mask_grid.shape[1]))
        
        # result_array = np.vstack([result_array, mask_grid[None, :, :]])
        
        
        return i, mask_grid
    
    num_iterations = len(sh)
    # Create a ThreadPoolExecutor
    start = time.time()
    with Pool() as pool:
    # Pass a range of values to the pool, each value representing an iteration
        # results = pool.map(para_proc, range(num_iterations))
        results = list(tqdm(pool.imap_unordered(para_proc, range(num_iterations)), total=num_iterations))

    pool.close()
    pool.join()
    
    print('Finished in ', round( ( time.time() - start ),0 ), ' seconds')
    indices, results = zip(*results)
    
    new_combined_array = np.stack(results, axis=0)
    
    # results.sort(key=lambda x: x[1])
    # # Extract the arrays from the sorted results
    # sorted_arrays = [result[0] for result in results]
    
    # # Concatenate the arrays to form the final result
    # result_array = np.stack(sorted_arrays, axis=0)

'''
Eventually add this back into an nc file for later processing, could stick all the data 
that is to be compared together? It would make processing results easier? Although maybe keep U,V and T seperated ? or
combine them but keep their grids seperated? 

plotting = 'n'
if plotting == 'y':
    plt.figure()
    plt.pcolor(lons_sliced, lats_sliced, interpolated_values_sliced,linewidth=0,rasterized=True)
    plt.colorbar()
    
    plt.figure()
    plt.pcolor(masked,linewidth=0,rasterized=True)
    plt.colorbar()
    plt.title('UKC4 example')
    
    plt.figure()
    plt.pcolor(lons_sliced, lats_sliced, mask_grid,linewidth=0,rasterized=True)
    plt.colorbar()
    plt.title('PRIMEA regridded onto UKC4 example')
    
'''