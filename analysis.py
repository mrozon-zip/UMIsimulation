import os
import csv

def count_rows(file_path):
    """Return the number of rows in the given CSV file."""
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        return sum(1 for _ in reader)

def average_rows_excluding_mut(folder_path):
    """
    Calculate the average number of rows for CSV files in the folder that
    do not have 'mut_0_' in their filename.
    """
    total_rows = 0
    file_count = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv") and "mut_0_" not in filename:
            file_path = os.path.join(folder_path, filename)
            rows = count_rows(file_path)
            total_rows += rows
            file_count += 1
    return total_rows / file_count if file_count > 0 else 0

# Define the folder path
folder_path = "results"

# Print the number of rows for each CSV file
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        rows = count_rows(file_path)
        print(f"{filename} : {rows}")

# Calculate and print the average row count for files that don't include "mut_0_" in their name
avg_rows = average_rows_excluding_mut(folder_path)
print(f"\nAverage rows for files without 'mut_0_' in their name: {avg_rows}")
