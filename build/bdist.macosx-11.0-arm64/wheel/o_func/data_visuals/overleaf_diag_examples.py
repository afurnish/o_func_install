# Adjusting the wavy water surface to be correctly placed along the top horizontal edge

import numpy as np
import matplotlib.pyplot as plt

# Data for the Law of the Wall (1/7th Power Law)
y = np.linspace(0, 1, 100)  # normalized distance from the boundary (0 = boundary, 1 = far from boundary)
u = y**(1/7)  # velocity profile according to the 1/7th power law

# Creating a wavy water surface along the top
x_wave = np.linspace(0, 1, 100)
wave_amplitude = 0.05
wave = wave_amplitude * np.sin(4 * np.pi * x_wave) + 1

# Plotting the velocity profile
plt.figure(figsize=(6, 8))
plt.plot(u, y, color='black', label=r'$\propto y^{1/7}$')

# Labels and grid
plt.xlabel('Current Velocity')
plt.ylabel('Height Above the Bed')
plt.grid(True, linewidth=0.5)

# Add seabed
# plt.fill_betweenx(y, u, 0, where=(y == y[0]), color='brown', alpha=0.3, label='Seabed')
plt.axhline(0, color='black', linewidth=1)

# Add viscous sublayer
plt.fill_betweenx(y, u, 0, where=(y <= 0.1), color='yellow', alpha=0.3, label='Viscous Sublayer')
# plt.text(0.4, 0.05, 'Viscous Sublayer', verticalalignment='center')

# Adding wavy water surface
plt.plot(x_wave, wave, color='blue', linewidth=2, label='Water Surface')
# plt.fill_between(x_wave, wave, wave + 0.1, color='blue', alpha=0.3)

# Invert the x-axis to reflect the correct direction
plt.gca().invert_xaxis()

# Set legend position to top right
plt.legend(loc='upper right')

plt.savefig('law_of_the_wall_plot.png', dpi=300, bbox_inches='tight')

# Show plot
plt.show()

#%% 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Sample data: event-mean variability and tidal range (similar to your image)
event_mean_variability = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
tidal_range = np.array([15, 12, 10, 8, 6, 5, 4, 3, 2, 1])
non_tidal_ratio = np.array([10, 12, 15, 18, 22, 28, 35, 45, 55, 70])

# Define a model for the best fit curve
def model_func(x, a, b):
    return a * np.exp(b * x)

# Fit the model to the data
popt_tidal, _ = curve_fit(model_func, event_mean_variability, tidal_range)
popt_non_tidal, _ = curve_fit(model_func, event_mean_variability, non_tidal_ratio)

# Generate fitted data
fitted_tidal = model_func(event_mean_variability, *popt_tidal)
fitted_non_tidal = model_func(event_mean_variability, *popt_non_tidal)

# Create the first plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(event_mean_variability, tidal_range, color='blue')
plt.plot(event_mean_variability, fitted_tidal, color='black')
plt.xlabel('Event-mean variability (%)')
plt.ylabel('Tidal range (m)')
plt.title('Event-mean variability plotted against tidal range')
plt.text(8, 10, r'$R^2 = 0.78$', fontsize=12)

# Create the second plot
plt.subplot(1, 2, 2)
plt.scatter(event_mean_variability, non_tidal_ratio, color='blue')
plt.plot(event_mean_variability, fitted_non_tidal, color='black')
plt.xlabel('Event-mean variability (%)')
plt.ylabel('Non-tidal residual: tidal range ratio (%)')
plt.title('Event-mean variability plotted against non-tidal residual: tidal range ratio')
plt.text(8, 40, r'$R^2 = 0.86$', fontsize=12)

plt.tight_layout()
plt.savefig('event_mean_probability.png', dpi=300, bbox_inches='tight')

plt.show()
