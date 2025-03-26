import os

def prepend_prefix_to_csv_files(prefix="polonies_", folder="results1"):
    """
    Rename all .csv files in the specified folder by adding a prefix to their filename.

    Parameters:
        prefix (str): The prefix to add to each .csv file.
        folder (str): The folder in which to search for .csv files. Defaults to the current directory.
    """
    # List all files in the folder
    for filename in os.listdir(folder):
        # Only consider .csv files
        if filename.endswith(".csv"):
            # Create the new filename by prepending the prefix
            new_filename = prefix + filename
            # Build full file paths
            old_file = os.path.join(folder, filename)
            new_file = os.path.join(folder, new_filename)

            # Rename the file
            os.rename(old_file, new_file)
            print(f"Renamed: {filename} -> {new_filename}")


if __name__ == '__main__':
    # Call the function with the current folder (or change to your target folder)
    prepend_prefix_to_csv_files()