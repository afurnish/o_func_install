import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ttide import t_tide, t_predic

# 1. Create synthetic time and signal (M2 + S2 + noise)
np.random.seed(0)
hours = 2881
t = pd.date_range(start='2013-10-31 01:30:00', periods=hours, freq='H')
# signal = (
#     1.5 * np.sin(2 * np.pi * (1 / 12.42) * np.arange(hours)) )
freq_M2 = 1 / 12.42
freq_S2 = 1 / 12.00
hours = 2881
t_hours = np.arange(hours)
# Construct the signal: M2 + S2 + noise
signal = (
    1.5 * np.sin(2 * np.pi * freq_M2 * t_hours) +        # M2
    0.8 * np.sin(2 * np.pi * freq_S2 * t_hours + 0.5) +  # S2 with a phase shift
    0.2 * np.random.randn(hours)                         # Additive Gaussian noise
)
# 2. Tidal analysis (same start time as signal)
print('--------------------------------t_tide')

analysis = t_tide(signal, dt=1, stime=t[0].to_pydatetime(), lat=53.5, constitnames=['M2', 'S2'])

# 3. Predict tide using t_predic with identical time array
print('------------------------------t_predic')
predicted = t_predic(
    t_time=t.to_pydatetime(),
    tidecon=analysis['tidecon'],
    names=analysis['nameu'],
    freq=analysis['fu'],
    lat=53.5
)

# 4. Plot for comparison
plt.figure(figsize=(15, 5))
plt.plot(t, signal, color='black', label='Synthetic observed')
plt.plot(t, predicted, color='blue', alpha=0.7, label='Predicted (t_predic)')
plt.title('Synthetic Tide: Observed vs Predicted')
plt.xlabel('Date')
# %%
plt.ylabel('Elevation')

plt.legend()
plt.tight_layout()
plt.show()

#%% 