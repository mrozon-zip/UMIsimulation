import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import ast


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
            active_field = row['active'].strip()
            born_field = row['born'].strip()

            # Parse born values
            if born_field.startswith('[') and born_field.endswith(']'):
                try:
                    born_list = ast.literal_eval(born_field)
                except (ValueError, SyntaxError) as e:
                    print(f"Skipping row {row} due to born conversion error: {e}")
                    continue
            else:
                try:
                    born_list = [int(born_field)]
                except ValueError as e:
                    print(f"Skipping row {row} due to born conversion error: {e}")
                    continue

            # Parse active values
            if active_field.startswith('[') and active_field.endswith(']'):
                try:
                    active_list = ast.literal_eval(active_field)
                except (ValueError, SyntaxError) as e:
                    print(f"Skipping row {row} due to active list parse error: {e}")
                    continue
            else:
                try:
                    active_list = [int(active_field)]
                except ValueError as e:
                    print(f"Skipping row {row} due to active conversion error: {e}")
                    continue

            # Append all combinations of born and active values
            for born_val in born_list:
                try:
                    born_int = int(born_val)
                except ValueError as e:
                    print(f"Skipping born value {born_val} due to conversion error: {e}")
                    continue
                for act in active_list:
                    try:
                        active_int = int(act)
                    except ValueError as e:
                        print(f"Skipping active value {act} due to conversion error: {e}")
                        continue
                    data.append({'active': active_int, 'born': born_int})
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
    parser = argparse.ArgumentParser(description='Create heatmap from multiple CSV files.')
    parser.add_argument('preset', choices=['pcr', 'polonies'], help='Preset to select files.')
    parser.add_argument('--input-dir', default='results_amplified', help='Directory containing CSV files.')
    parser.add_argument('--output', help='Output filename for the heatmap image.')
    args = parser.parse_args()

    pattern = f"*{args.preset}*.csv"
    file_paths = glob.glob(os.path.join(args.input_dir, pattern))
    if not file_paths:
        raise FileNotFoundError(f"No files matching pattern {pattern} in directory {args.input_dir}")

    all_data = []
    for file_path in file_paths:
        all_data.extend(load_csv_to_dicts(file_path))

    output_filename = args.output or f"heatmap_{args.preset}.png"
    create_heatmap(all_data, output_filename)