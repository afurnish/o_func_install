import xarray as xr
import numpy as np

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