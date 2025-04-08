import csv
import os
import matplotlib.pyplot as plt
import numpy as np


def load_csv_to_dicts(file_path):
    """
    Loads the CSV file and returns a list of dictionaries containing only 'active' and 'born' keys,
    converting their values to integers.
    Raises FileNotFoundError if the file is not found,
    and ValueError if the required headers are missing.
    """
    data = []
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found.")

    with open(file_path, mode='r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        # Check that required columns exist
        if 'active' not in reader.fieldnames or 'born' not in reader.fieldnames:
            raise ValueError("CSV file must contain 'active' and 'born' columns.")

        for row in reader:
            try:
                active_val = int(row['active'])
                born_val = int(row['born'])
            except ValueError as e:
                # Skip rows that cannot be converted to integers
                print(f"Skipping row {row} due to conversion error: {e}")
                continue
            data.append({'active': active_val, 'born': born_val})
    return data


def create_heatmap(data, output_filename):
    """
    Creates and displays a heatmap that shows the normalized frequency of molecules per active cycle count,
    grouped by birth cycle. The normalization is done for each birth cycle (row).
    The resulting heatmap is saved as a PNG file.
    """
    if not data:
        raise ValueError("No data provided to create heatmap.")

    # Organize data: dictionary with key as 'born' and inner dictionary counting 'active' values.
    frequency = {}
    for entry in data:
        born = entry['born']
        active = entry['active']
        if born not in frequency:
            frequency[born] = {}
        frequency[born][active] = frequency[born].get(active, 0) + 1

    # Identify unique birth and active cycle counts, sorted for consistency.
    born_values = sorted(frequency.keys())
    active_values = sorted({active for counts in frequency.values() for active in counts.keys()})

    # Prepare a matrix to hold normalized frequencies.
    heatmap_matrix = np.zeros((len(born_values), len(active_values)))

    # Fill matrix: for each birth cycle, normalize counts of active cycles.
    for i, born in enumerate(born_values):
        row_counts = frequency[born]
        total = sum(row_counts.values())
        for j, active in enumerate(active_values):
            count = row_counts.get(active, 0)
            heatmap_matrix[i, j] = count / total if total > 0 else 0

    # Plotting the heatmap using matplotlib
    plt.figure(figsize=(8, 6))
    img = plt.imshow(heatmap_matrix, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(img, label='Normalized Frequency')
    plt.xlabel('Active Cycle Count')
    plt.ylabel('Birth Cycle')

    # Set ticks to show the actual active and born values
    plt.xticks(ticks=np.arange(len(active_values)), labels=active_values)
    plt.yticks(ticks=np.arange(len(born_values)), labels=born_values)
    plt.title('Heatmap of Normalized Active Cycle Distribution per Birth Cycle')
    plt.tight_layout()

    # Save the heatmap as a PNG file and display it.
    plt.savefig(output_filename)
    print(f"Heatmap saved as '{output_filename}'")
    plt.show()

if __name__ == '__main__':
    data = load_csv_to_dicts('results1/pcr_amplified.csv')
    create_heatmap(data, 'heatmap_pcr.png')