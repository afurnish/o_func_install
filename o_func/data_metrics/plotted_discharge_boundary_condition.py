import pandas as pd
from datetime import datetime, timedelta

def read_bc_timeseries(filepath, name="Mersey_0001"):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    collecting = False
    base_time = None
    time_discharge = []

    for line in lines:
        line = line.strip()

        if line.startswith("[forcing]"):
            collecting = False  # Reset in case we're between blocks
            continue

        if line.lower().startswith("name") and name in line:
            collecting = True
            continue

        if collecting:
            if line.lower().startswith("unit") and "since" in line:
                # Parse base time
                base_str = line.split("since")[1].strip()
                base_time = datetime.strptime(base_str, "%Y-%m-%d %H:%M:%S")
                continue

            # Only collect lines with two numeric values
            parts = line.split()
            if len(parts) == 2:
                try:
                    t = float(parts[0])
                    q = float(parts[1])
                    time_discharge.append((t, q))
                except ValueError:
                    continue  # Ignore bad lines

    if not time_discharge or base_time is None:
        raise ValueError("Could not find time series or base time in the file.")

    # Convert seconds since base_time to actual datetimes
    datetimes = [base_time + timedelta(seconds=t) for t, _ in time_discharge]
    discharges = [q for _, q in time_discharge]

    df = pd.DataFrame({
        "datetime": datetimes,
        "discharge_m3s": discharges
    })

    return df

df = read_bc_timeseries("/Volumes/PNC/modelling_DATA/kent_estuary_project/12.salinity_calibration_laststeps/models/ao_nawind_AllRivNoDuddonClimatology_m0.035_Forcing/runSCW_ao_nawind_AllRivNoDuddonClimatology_m0.035_Forcing/Discharge.bc", name="Mersey_0001")

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.plot(df['datetime'], df['discharge_m3s'], label="Mersey from .bc file")
plt.xlabel("Date")
plt.ylabel("Discharge [m³/s]")
plt.title("Mersey Discharge from Forcing File")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()

df = read_bc_timeseries('/Volumes/PNC/modelling_DATA/kent_estuary_project/12.salinity_calibration_laststeps/models/ao_yawind_orig8RealRiver_m0.035_Forcing_85_Discouv/runSCW_ao_yawind_orig8RealRiver_m0.035_Forcing_85_Discouv/Discharge.bc", name="Mersey_0001)
)