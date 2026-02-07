#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 12:33:08 2025

@author: af
"""
#%%
# Compute residual
residual = np.array(sh_data_at_point) - predicted_tide

# Plot everything
plt.figure(figsize=(15, 6))
# plt.plot(tt_time_py, sh_data_at_point, label='Observed', color='black')
# plt.plot(tt_time_py, predicted_tide, label='Predicted Tide', color='blue')
plt.plot(tt_time_py, residual, label='Residual (Storm Surge)', color='red')

plt.title("Tide Decomposition at Point ({}, {})".format(point[0], point[1]))
plt.xlabel("Date")
plt.ylabel("Water Level (m)")
plt.legend()
plt.grid(True)

# Zoom to first 14 days
# plt.xlim(t_datetime[0], t_datetime[0] + timedelta(days=14))

# Optional: tidy date format
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%% 
mdict = {
    "h": np.array(sh_data_at_point.values),  # Your water levels
    "time": [str(t) for t in pd.to_datetime(sh_data_at_point.time_primea.values)]  # ISO 8601 strings
}
savemat("test_tide.mat", mdict)

sh_data_at_point

path = '/home/af/Documents/MATLAB/matlab_predicted_surface.csv'




# Load MATLAB prediction
matlab_df = pd.read_csv(path, header=None)
matlab_pred = matlab_df.iloc[:, -1].values  # Assuming last column is predicted surface height

# Python prediction (already computed)
# predicted_tide = your result from tt.t_predic(...)
# Align lengths if needed
min_len = min(len(matlab_pred), len(predicted_tide))
diff = matlab_pred[:min_len] - predicted_tide[:min_len]

print("Mean difference:", np.mean(diff))
print("Max difference:", np.max(np.abs(diff)))

plt.figure()
plt.plot(predicted_tide)
plt.plot(matlab_pred)






