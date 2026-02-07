# import xarray as xr
# import numpy as np
# import matplotlib.pyplot as plt

# # Path to NetCDF file
# file_path = "/Volumes/PNC/Original_Data/UKC3/river_climatology/rivers/AMM15_River_Climatology_v2.nc"

# # Load dataset and extract first time slice
# ds = xr.open_dataset(file_path)
# rorunoff = ds['rorunoff'].isel(time_counter=0)

# # Create a mask for values > 0 (actual river cells)
# river_mask = rorunoff > 0
# masked_river = np.ma.masked_where(~river_mask, river_mask)

# # Plot: red river cells on white background
# plt.figure(figsize=(10, 8))
# plt.imshow(masked_river, cmap='Reds', origin='lower')
# plt.gca().set_facecolor("white")
# plt.title("River Grid Cells (rorunoff > 0)")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.colorbar(label="River Presence (1 = river)")
# plt.tight_layout()
# plt.show()


# print(f"Number of river cells with rorunoff > 0: {river_mask.sum().item()}")
# #%% 
# import xarray as xr
# import numpy as np
# import geopandas as gpd
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature

# # Load river data
# file_path = "/Volumes/PNC/Original_Data/UKC3/river_climatology/rivers/AMM15_River_Climatology_v2.nc"
# ds = xr.open_dataset(file_path)
# rorunoff = ds['rorunoff'].isel(time_counter=0)

# # Create mask of river points
# river_mask = rorunoff > 0
# river_y, river_x = np.where(river_mask.values)

# # Extract lat/lon
# lat = ds['lat'].values
# lon = ds['lon'].values
# river_lats = lat[river_y, river_x]
# river_lons = lon[river_y, river_x]

# # Create GeoDataFrame
# gdf_rivers = gpd.GeoDataFrame(geometry=gpd.points_from_xy(river_lons, river_lats), crs="EPSG:4326")

# # Load UK shapefile manually (update path to where you extract the .shp)
# shapefile_path = "/Volumes/PNC/Original_Data/UKC3/river_climatology/rivers/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp"
# world = gpd.read_file(shapefile_path)
# uk = world[world['NAME'] == 'United Kingdom']

# # Plot
# fig = plt.figure(figsize=(12, 14))
# ax = plt.axes(projection=ccrs.PlateCarree())
# ax.set_extent([-11, 2, 49.5, 61], crs=ccrs.PlateCarree())

# uk.plot(ax=ax, facecolor='lightgrey', edgecolor='black')
# ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
# gdf_rivers.plot(ax=ax, color='red', markersize=10, zorder=5)

# plt.title("River Grid Cells (rorunoff > 0) on High-Resolution UK Map")
# plt.tight_layout()
# plt.show()

#%%
import xarray as xr
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.ticker import FormatStrFormatter

# === Load NetCDF river grid ===
ds = xr.open_dataset("/Volumes/PNC/Original_Data/UKC3/river_climatology/rivers/AMM15_River_Climatology_v2.nc")
rorunoff = ds['rorunoff'].isel(time_counter=0)
river_mask = rorunoff > 0
river_y, river_x = np.where(river_mask.values)

lat = ds['lat'].values
lon = ds['lon'].values
river_lats = lat[river_y, river_x]
river_lons = lon[river_y, river_x]

gdf_rivers = gpd.GeoDataFrame(
    geometry=gpd.points_from_xy(river_lons, river_lats),
    crs="EPSG:4326"
)

# === Load UK shapefile ===
shapefile_path = "/Volumes/PNC/Original_Data/UKC3/river_climatology/rivers/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp"
world = gpd.read_file(shapefile_path)
uk = world[world['NAME'] == 'United Kingdom']

# === Plot ===
fig = plt.figure(figsize=(5, 7))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-11, 2, 49.5, 61], crs=ccrs.PlateCarree())

# Plot UK land and river discharge points
uk.boundary.plot(ax=ax, edgecolor='black', linewidth=1, zorder=1)
gdf_rivers.plot(ax=ax, color='red', markersize=10, label="Discharge Locations", zorder=2)

# Coastline
ax.add_feature(cfeature.COASTLINE.with_scale('10m'))

# Gridlines with decimal degree formatting
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xformatter = FormatStrFormatter('%.0f°')
gl.yformatter = FormatStrFormatter('%.0f°')
gl.xlabel_style = {'size': 10}
gl.ylabel_style = {'size': 10}

# Axis labels using anchored text
ax.text(0.5, -0.05, 'Longitude', transform=ax.transAxes,
        ha='center', va='top', fontsize=12)
ax.text(-0.13, 0.5, 'Latitude', transform=ax.transAxes,
        ha='center', va='center', rotation='vertical', fontsize=12)

# Add legend
ax.legend(loc='upper left', fontsize=11, frameon=True)

plt.tight_layout()
plt.savefig('River_location_discharge_ukc4.png', dpi = 300)