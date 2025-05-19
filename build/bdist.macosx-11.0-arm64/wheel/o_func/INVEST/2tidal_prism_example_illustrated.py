import numpy as np

import matplotlib.pyplot as plt



# Simulated data for Q_tidal (tidal discharge in m^3/hour)

time_hours = np.linspace(0, 24, 1000)  # 24-hour period sampled 1000 times

Q_tidal = 50000 * np.sin(2 * np.pi * time_hours / 12)  # Sinusoidal tidal flux



# Calculate absolute tidal flux for integration

Q_tidal_abs = np.abs(Q_tidal)



# Integration using trapezoidal rule to approximate tidal prism

tidal_prism = np.trapz(Q_tidal_abs, x=time_hours)


# Plotting the tidal flux with shaded areas for positive and negative contributions
plt.figure(figsize=(10, 6))

# Plot the tidal flux
plt.plot(time_hours, Q_tidal, label="Tidal Flux (Q_tidal)", color="blue")

# Shade positive (flood) area
plt.fill_between(time_hours, 0, Q_tidal, where=Q_tidal > 0, 
                 color="green", alpha=0.5, label="Flood (Positive Area)")

# Shade negative (ebb) area
plt.fill_between(time_hours, 0, Q_tidal, where=Q_tidal < 0, 
                 color="red", alpha=0.5, label="Ebb (Negative Area)")

# Add a zero line
plt.axhline(0, color="black", linestyle="--", linewidth=0.8, label="Zero Line")

# Add labels and legend
plt.title("Tidal Flux with Positive (Flood) and Negative (Ebb) Contributions")
plt.xlabel("Time (hours)")
plt.ylabel("Tidal Flux (m³/hour)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()
