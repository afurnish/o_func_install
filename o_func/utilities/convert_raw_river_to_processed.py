import os
import pandas as pd

# Define directories
unprocessed_dir = '/media/af/PN/modelling_DATA/kent_estuary_project/river_boundary_conditions/delft_bc_files/extra_rivers_real_data_esk_alt_clywd/unprocessed'
processed_dir = '/media/af/PN/modelling_DATA/kent_estuary_project/river_boundary_conditions/delft_bc_files/extra_rivers_real_data_esk_alt_clywd/processed'

# Ensure the processed directory exists
os.makedirs(processed_dir, exist_ok=True)

# Function to process and validate CAMELS files
def process_camels_file(file_path):
    df = pd.read_csv(file_path, encoding='latin1')
    if 'discharge_vol' in df.columns:
        processed_df = df[['date', 'discharge_vol']].dropna()
        processed_df['date'] = pd.to_datetime(processed_df['date'], errors='coerce')

        if processed_df['date'].isna().any():
            print(f"Warning: Invalid dates found in {file_path}.")
        if not pd.to_numeric(processed_df['discharge_vol'], errors='coerce').notna().all():
            print(f"Warning: Non-numeric discharge values found in {file_path}.")
        return processed_df[['date', 'discharge_vol']]

# Function to process and validate NR files
def process_nr_file(file_path):
    names = ['Time_stamp', 'Value', 'State_of_value', 'Interpolation', 'Tags', 'Comments']
    df = pd.read_csv(file_path, header=17, encoding='latin1', names=names)
    df['Time_stamp'] = pd.to_datetime(df['Time_stamp'], format='%d/%m/%Y %H:%M:%S', errors='coerce')

    if df['Time_stamp'].isna().any():
        print(f"Warning: Invalid dates found in {file_path}.")
    if not pd.to_numeric(df['Value'], errors='coerce').notna().all():
        print(f"Warning: Non-numeric discharge values found in {file_path}.")
    return df[['Time_stamp', 'Value']].dropna()

# Function to check for missing time steps
def check_missing_time_steps(df, time_column, file_path):
    time_intervals = df[time_column].diff().dropna()
    avg_interval = time_intervals.mode()[0]
    expected_times = pd.date_range(start=df[time_column].min(), end=df[time_column].max(), freq=avg_interval)
    missing_times = expected_times.difference(df[time_column])

    if not missing_times.empty:
        print(f"Warning: Missing time steps detected in {file_path}.")
        print(f"Missing time steps: {missing_times}")

# Main function to process all files in the unprocessed directory
if __name__ == "__main__":
    for filename in os.listdir(unprocessed_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(unprocessed_dir, filename)
            river_name = filename.split('_')[0]

            if "CAMELS" in filename:
                processed_df = process_camels_file(file_path)
                if processed_df is not None:
                    check_missing_time_steps(processed_df, 'date', file_path)
                    output_path = os.path.join(processed_dir, f"{river_name}_.csv")
                    processed_df.to_csv(output_path, index=False, header=False)
            else:
                processed_df = process_nr_file(file_path)
                if processed_df is not None:
                    check_missing_time_steps(processed_df, 'Time_stamp', file_path)
                    output_path = os.path.join(processed_dir, f"{river_name}_.csv")
                    processed_df.to_csv(output_path, index=False, header=False)

    print("Data processing and validation complete.")
