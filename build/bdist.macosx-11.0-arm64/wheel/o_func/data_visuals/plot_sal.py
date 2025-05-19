import xarray as xr
import matplotlib.pyplot as plt


#Plot unstructured
import matplotlib.animation as animation

dataset = xr.open_dataset('C:/Users/aafur/Downloads/kent_31_merged_map.nc')
salinity = dataset.mesh2d_sa1

x = dataset.mesh2d_face_x.values
y = dataset.mesh2d_face_y.values
z = salinity[12,:]
plt.scatter(x, y, c = z, s = 0.5)


fig, ax = plt.subplots(figsize=(10, 8))
scat = ax.scatter(x, y, c=salinity[0], s=0.5, cmap='viridis')
fig.colorbar(scat, ax=ax)
ax.set_title('Salinity - Initial timestep')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Function to update the plot for each frame
def update(frame):
    scat.set_array(salinity[frame].values)
    ax.set_title(f'Salinity - Time {frame+1}')
    return scat,

# Create and save the animation
ani = animation.FuncAnimation(fig, update, frames=len(salinity.time), repeat=False)
ani.save('unstructured_salinity_animation.mp4', writer='ffmpeg', fps=2)




#Plot structured. 
data = xr.open_dataset('C:/Users/aafur/Downloads/kent_regrid.nc')

prim_last_salinity = data['prim_surface_salinity'][8]
ukc4_last_salinity = data['ukc4_surface_salinity'][8]

# Plotting
fig, axes = plt.subplots(ncols=2, figsize=(14, 6), sharey=True)

# Prim salinity plot
cmap1 = axes[0].pcolormesh(prim_last_salinity, cmap='viridis')
fig.colorbar(cmap1, ax=axes[0])
axes[0].set_title('Prim Surface Salinity (last timestep)')
axes[0].set_xlabel('Longitude index')
axes[0].set_ylabel('Latitude index')

# UKC4 salinity plot
cmap2 = axes[1].pcolormesh(ukc4_last_salinity, cmap='viridis')
fig.colorbar(cmap2, ax=axes[1])
axes[1].set_title('UKC4 Surface Salinity (last timestep)')
axes[1].set_xlabel('Longitude index')