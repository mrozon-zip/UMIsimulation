import os
import argparse
import csv
import ast
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

csv.field_size_limit(30 * 1024 * 1024)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a locally-normalized heatmap of active vs. birth cycles "
                    "from PCR/Polonies data files.")
    parser.add_argument(
        "folder",
        help="Path to the folder containing your data files")
    parser.add_argument(
        "--pattern", "-p",
        action='append',
        required=True,
        help="Substring(s) to match in filenames (repeat for multiple patterns, e.g., -p polonies_mut_0.0 -p AOE)")
    return parser.parse_args()

def find_files(folder, patterns):
    """Return all file paths in `folder` whose names contain ALL substrings in `patterns`."""
    return [
        os.path.join(folder, fname)
        for fname in os.listdir(folder)
        if all(pat in fname for pat in patterns)
    ]

def process_files(file_list):
    """
    Parse each CSV, extract born/active lists, and count occurrences of each (born, active) pair.
    Returns:
      - counts: dict mapping (born, active) → count
      - sorted list of unique born values
      - sorted list of unique active values
    """
    counts = defaultdict(int)
    born_vals = set()
    active_vals = set()

    for path in file_list:
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                born_list = ast.literal_eval(row["born"])
                active_list = ast.literal_eval(row["active"])
                for b, a in zip(born_list, active_list):
                    counts[(b, a)] += 1
                    born_vals.add(b)
                    active_vals.add(a)

    return counts, sorted(born_vals), sorted(active_vals)

def build_matrix(counts, born_values, active_values):
    """
    Build a 2D NumPy array of shape (len(born_values), len(active_values)),
    fill it with counts, then normalize each row so it sums to 1.
    """
    matrix = np.zeros((len(born_values), len(active_values)), dtype=float)
    b_index = {b: i for i, b in enumerate(born_values)}
    a_index = {a: j for j, a in enumerate(active_values)}

    for (b, a), cnt in counts.items():
        matrix[b_index[b], a_index[a]] = cnt

    # Global normalization
    total = matrix.sum()
    if total == 0:
        total = 1  # avoid division by zero
    matrix /= total

    return matrix

def plot_heatmap(matrix, born_values, active_values, output_filename):
    """Render the heatmap with birth cycles on Y and active cycles on X."""
    plt.figure(figsize=(10, 8))
    im = plt.imshow(matrix, origin='lower', aspect='auto')
    plt.colorbar(im, label='Proportion (row-normalized)')
    plt.xticks(ticks=np.arange(len(active_values)), labels=active_values, rotation=90)
    plt.yticks(ticks=np.arange(len(born_values)), labels=born_values)
    plt.xlabel("Active cycle count")
    plt.ylabel("Birth cycle")
    plt.title("Heatmap of Active vs. Birth Cycles (Globaly Normalized)")
    plt.tight_layout()
    plt.savefig(output_filename, format='svg')
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    folder = args.folder
    patterns = args.pattern
    files = find_files(folder, patterns)
    if not files:
        print(f"No files found in “{folder}” matching “{args.pattern}”.")

    counts, born_vals, active_vals = process_files(files)
    matrix = build_matrix(counts, born_vals, active_vals)
    # Export raw counts using counts dict to CSV
    counts_csv_filename = '_'.join(patterns) + '_heatmap_counts.csv'
    with open(counts_csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # header row with birth_cycle and active_vals
        writer.writerow(['birth_cycle'] + active_vals)
        # write each row using the counts dict
        for b in born_vals:
            row = [counts.get((b, a), 0) for a in active_vals]
            writer.writerow([b] + row)
    output_filename = '_'.join(patterns) + '_heatmap.svg'
    plot_heatmap(matrix, born_vals, active_vals, output_filename)