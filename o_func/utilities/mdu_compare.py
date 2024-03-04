import os
import glob
from o_func import opsys  # Ensure this module and function are correctly defined elsewhere

def list_model_directories(main_folder_path):
    # List all model directories within the specified path
    model_dirs = [d for d in glob.glob(os.path.join(main_folder_path, '**', 'runSCW*'), recursive=True) if os.path.isdir(d)]
    return model_dirs

def user_select_model_directories(model_dirs):
    # Print all model directories for user selection, but only show the last name in the path
    print("Available model directories:")
    for i, model_dir in enumerate(model_dirs):
        # Extract the last name from the model directory path
        last_name = os.path.basename(model_dir)
        print(f"{i}: {last_name}")
    
    # Allow user to select multiple directories by their indices
    selected_indices = input("Enter the indices of directories to compare, separated by commas (e.g., 0,1,2): ")
    selected_indices = [int(idx.strip()) for idx in selected_indices.split(',') if idx.strip().isdigit()]
    
    return [model_dirs[i] for i in selected_indices if i < len(model_dirs)]

def find_mdu_files(model_dirs):
    mdu_files = []
    for model_dir in model_dirs:
        # Adjust the pattern to search within 'runSCW*' subdirectories
        pattern = os.path.join(model_dir, '*.mdu')
        found_files = glob.glob(pattern, recursive=True)
        print(f"Searching in: {pattern}")  # Debugging output
        if found_files:
            print(f"Found .mdu files: {found_files}")  # Debugging output
        else:
            print("No .mdu files found in this subdirectory.")  # Debugging output
        mdu_files.extend(found_files)
    return mdu_files


def compare_files(mdu_files):
    if len(mdu_files) < 2:
        print("Need at least two files to compare.")
        return
    
    # Read contents of each mdu file
    contents = [read_mdu_file(fp) for fp in mdu_files]
    file_contents = {fp: cnt for fp, cnt in zip(mdu_files, contents)}
    
    # Extract base file for comparison
    base_fp = mdu_files[0]
    base_contents = file_contents[base_fp]
    print(f"Base file for comparison: {os.path.basename(base_fp)}")
    
    # Compare each file against the base
    for fp in mdu_files[1:]:
        compare_contents = file_contents[fp]
        print(f"\nComparing with: {os.path.basename(fp)}")
        differences = False
        
        # Compare line by line, considering the longest file
        max_len = max(len(base_contents), len(compare_contents))
        for i in range(max_len):
            base_line = base_contents[i] if i < len(base_contents) else "N/A"
            compare_line = compare_contents[i] if i < len(compare_contents) else "N/A"
            if base_line != compare_line:
                print(f"Difference at line {i+1}:")
                print(f"  Base: {base_line}")
                print(f"  Compare: {compare_line}")
                differences = True
        
        if not differences:
            print("No differences found.")
        
        # Print file paths clearly indicating which one is base and which one is compare
        print(f"\nBase file path: {base_fp}")
        print(f"Compared file path: {fp}")


def read_mdu_file(file_path):
    # Adjusted to return file path for identification
    with open(file_path, 'r') as file:
        content = file.read().splitlines()
        for i, line in enumerate(content):
            if line.strip() == "":
                return content[i+1:]  # Return content after junk
    return []


def main():
    start_path = opsys()  # Get start path from opsys function
    main_folder_path = os.path.join(start_path, 'modelling_DATA', 'kent_estuary_project', '7.met_office', 'models')
    model_dirs = list_model_directories(main_folder_path)
    selected_model_dirs = user_select_model_directories(model_dirs)
    mdu_files = find_mdu_files(selected_model_dirs)
    compare_files(mdu_files)
    # Optionally, print paths for the user to open in Vim or another editor
    for file_path in mdu_files:
        pass
        # print(file_path)

if __name__ == "__main__":
    main()
