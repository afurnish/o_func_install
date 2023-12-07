import xarray as xr
import numpy as np

grid_example = '/Volumes/PN/modelling_DATA/kent_estuary_project/6.Final2/models/02_kent_1.0.0_UM_wind/shortrunSCW_kent_1.0.0_UM_wind/UK_West+Duddon+Raven_+liv+ribble+wyre+ll+ul+leven_kent_1.1_net.nc'

data = xr.open_dataset(grid_example)

def is_counterclockwise(A, B, C):
    area = 0.5 * ((B - A) * (C - A) - (C - A) * (B - A))
    print(area)
    return area > 0

# Assuming data is your xarray DataArray
face_nodes = data['mesh2d_face_nodes'].values
contains_nan = np.any(np.isnan(face_nodes), axis=1)

# Separate triangles and squares
triangles = face_nodes[contains_nan]
squares = face_nodes[~contains_nan]

for i in range(len(triangles)):
    # Assuming x-y coordinates only
    A, B, C = triangles[i, 0], triangles[i, 1], triangles[i, 2]

    # Check counterclockwise condition
    triangle_orientation = is_counterclockwise(A, B, C)
    print(f'Triangle {i + 1}: Counterclockwise = {triangle_orientation}')
