#!/usr/bin/env python3
import csv
import os
import argparse
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

"""
A hamming distance distribution graph constructor. Produces a PDF file with hamming distances in selected files. Not 
used in a final version.
"""

def read_csv_file(filepath, sample_size, min_rows):
    """
    Read a CSV file and return a list of dictionaries such that:
      - The cumulative sum of N0 (from the "N0" column) is at least sample_size.
      - At least min_rows rows are included.

    The rows are randomly sampled.

    Parameters:
        filepath (str): Path to the CSV file.
        sample_size (float): Target cumulative N0 value.
        min_rows (int): Minimum number of rows to include.

    Returns:
        List[Dict]: A list of dictionaries corresponding to the sampled rows.

    Raises:
        ValueError: If the CSV does not contain enough data to meet both conditions.
    """
    # Read the CSV file.
    with open(filepath, 'r', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Shuffle rows to ensure random sampling.
    random.shuffle(rows)

    cumulative_n0 = 0.0
    sampled_data = []

    # Iterate over rows and accumulate until both conditions are satisfied.
    for row in rows:
        try:
            n0_value = float(row['N0'])
        except ValueError:
            # If conversion fails, skip the row.
            continue

        sampled_data.append(row)
        cumulative_n0 += n0_value

        # Check if both conditions are met.
        if cumulative_n0 >= sample_size and len(sampled_data) >= min_rows:
            break

    # If conditions are not met after exhausting all rows, raise an error.
    if cumulative_n0 < sample_size or len(sampled_data) < min_rows:
        print("WARNING! Insufficient data to meet the required cumulative N0 and minimum row count.")

    return sampled_data

def pad_sequences(data):
    """
    Pad each row's 'sequence' with '-' so that all sequences are
    the same length as the longest one.
    """
    max_len = max(len(row['sequence']) for row in data)
    for row in data:
        seq = row['sequence']
        if len(seq) < max_len:
            row['sequence'] = seq + '-' * (max_len - len(seq))
    return data

def hamming_distance(seq1, seq2):
    """Compute the Hamming distance between two equal-length sequences."""
    return sum(c1 != c2 for c1, c2 in zip(seq1, seq2))

def levenshtein_distance(s1, s2):
    """Compute the Levenshtein distance between two sequences."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def compute_distance_distribution(sequences, metric="hamming"):
    """
    Compute the weighted distribution of pairwise distances (Hamming or Levenshtein)
    using only unique comparisons between different rows.

    For each pair (i, j) with i < j, compute the distance 'd' (using the chosen metric)
    and the weight as the product of the two N0 values. The distribution dictionary is updated
    as follows:
      - If d is already present as a key, add the weight to its cumulative count.
      - If d is not present, add a new key with the weight as its value.

    Returns:
        distribution (dict): Mapping distance -> weighted count.
    """
    distribution = {0: 0}  # Starting dictionary with key 0.
    n = len(sequences)
    for i in range(n):
        for j in range(i + 1, n):
            if metric == "hamming":
                d = hamming_distance(sequences[i]['sequence'], sequences[j]['sequence'])
            elif metric == "levenshtein":
                d = levenshtein_distance(sequences[i]['sequence'], sequences[j]['sequence'])
            # Calculate the weight from the product of the N0 values.
            weight = int(sequences[i]['N0']) * int(sequences[j]['N0'])
            # Update the distribution dictionary.
            if d in distribution:
                distribution[d] += weight
            else:
                distribution[d] = weight
    return distribution

def create_distance_distribution_figure(distribution, base, metric, source_file):
    """
    Create two bar chart subplots for the distance distribution.
    Left subplot: y-axis shows raw counts.
    Right subplot: y-axis shows raw counts on a logarithmic scale.
    Both plots are horizontally aligned and annotated with the source file.
    """
    dists = sorted(distribution.keys())
    counts = [distribution[d] for d in dists]

    # Create a figure with two horizontally aligned subplots.
    fig, axs = plt.subplots(ncols=2, figsize=(12, 5))

    # Left subplot: raw counts.
    axs[0].bar(dists, counts, width=0.8, color='skyblue', edgecolor='black')
    axs[0].set_xlim(0, max(dists))
    axs[0].set_xlabel(f"{metric.capitalize()} Distance")
    axs[0].set_ylabel("Count")
    axs[0].set_title(f"{metric.capitalize()} Distance Distribution (Counts)\nFile: {os.path.basename(source_file)}")
    axs[0].grid(True)

    # Right subplot: raw counts on a logarithmic y-axis.
    axs[1].bar(dists, counts, width=0.8, color='skyblue', edgecolor='black')
    axs[1].set_xlim(0, max(dists))
    axs[1].set_xlabel(f"{metric.capitalize()} Distance")
    axs[1].set_ylabel("Count (Log Scale)")
    axs[1].set_title(f"{metric.capitalize()} Distance Distribution (Log Scale)\nFile: {os.path.basename(source_file)}")
    axs[1].set_yscale('log')
    axs[1].grid(True)

    fig.tight_layout()
    return fig


def extract_info_from_filename(filepath):
    """
    Extract information from the filename based on underscore-separated tokens.
    The filename (without extension) is expected to have the following structure:
      amplification_key1_value1_key2_value2_..._keyN_valueN

    - The first segment (before the first underscore) is always treated as the "amplification" value.
    - Every subsequent two tokens represent a key and its corresponding value.
    - If a key matches one of the known shortcuts, it is renamed according to the mapping:
         mut  -> mutation_rate
         Sr   -> s_radius
         dens -> density
         SP   -> success_probability
         dev  -> deviation
      Otherwise, the key is used as is.

    Examples:
      For "bridge_mut_0.01_Sr_5_dens_20_SP_0.85_dev_0.05.csv" the result is:
         {
             "amplification": "bridge",
             "mutation_rate": "0.01",
             "s_radius": "5",
             "density": "20",
             "success_probability": "0.85",
             "deviation": "0.05"
         }

      For a filename with only one pair after amplification, e.g. "bridge_mut_0.01.csv":
         {
             "amplification": "bridge",
             "mutation_rate": "0.01"
         }
    """
    base = os.path.basename(filepath)
    base_no_ext, _ = os.path.splitext(base)
    tokens = base_no_ext.split('_')

    if not tokens:
        raise ValueError("Filename must contain at least one segment for amplification.")

    # The first token is always the amplification value.
    info = {"amplification": tokens[0]}

    # Remaining tokens must come in pairs.
    if (len(tokens) - 1) % 2 != 0:
        raise ValueError("Filename segments after the first must be in key-value pairs.")

    # Mapping for known segment shortcuts.
    mapping = {
        "mut": "mutation_rate",
        "Sr": "s_radius",
        "dens": "density",
        "SP": "success_probability",
        "dev": "deviation"
    }

    # Process each key/value pair.
    for i in range(1, len(tokens), 2):
        key = tokens[i]
        value = tokens[i+1]
        new_key = mapping.get(key, key)
        info[new_key] = value

    return info


def main():
    parser = argparse.ArgumentParser(
        description="Compute distance distributions and produce plots for CSV files."
    )
    parser.add_argument("csv_files", nargs="+", help="List of CSV files to process")
    parser.add_argument("--metric", type=str, choices=["hamming", "levenshtein", "both"],
                        default="hamming",
                        help="Distance metric to use: 'hamming' (requires equal length), 'levenshtein', or 'both'.")
    parser.add_argument("--sample_size", type=int, default=100, help="Sample size for analysis")
    parser.add_argument("--minrows", type=int, default=0, help="Minimum number of rows to sample")
    args = parser.parse_args()

    # List to store tuples of (file number, N0 values)
    file_n0_data = []

    # Create one aggregated PDF file for all plots.
    aggregated_pdf = "all_plots.pdf"
    with PdfPages(aggregated_pdf) as pdf:
        # Process each CSV file with an incremental file number starting from 1.
        for file_number, filepath in enumerate(args.csv_files, start=1):
            print(f"Processing {filepath}...")
            data = read_csv_file(filepath, args.sample_size, args.minrows)
            base, _ = os.path.splitext(filepath)
            n0_values = [int(row['N0']) for row in data]
            file_n0_data.append((file_number, n0_values))

            # Extract file information from the filename.
            info = extract_info_from_filename(filepath)
            # Create an info page with the extracted information.
            fig_info = plt.figure(figsize=(8.5, 11))
            plt.axis('off')
            filename_base = os.path.basename(filepath).lower()
            print(filename_base)
            if filename_base.startswith("pcr"):
                # For PCR files, show only amplification, mutation rate, and cycles.
                info_text = (
                    f"Amplification type: {info.get('amplification', 'N/A')}\n"
                    f"Mutation rate: {info.get('mutation_rate', 'N/A')}\n"
                    f"Cycles: {info.get('cycles', 'N/A')}\n"
                    f"File number: {file_number}"
                )
            else:
                # For polonies files, show all the extracted details.
                info_text = (
                    f"Amplification type: {info.get('amplification', 'N/A')}\n"
                    f"Mutation rate: {info.get('mutation_rate', 'N/A')}\n"
                    f"S_radius: {info.get('s_radius', 'N/A')}\n"
                    f"Density: {info.get('density', 'N/A')}\n"
                    f"Success Probability: {info.get('success_probability', 'N/A')}\n"
                    f"Deviation: {info.get('deviation', 'N/A')}\n"
                    f"File number: {file_number}"
                )
            plt.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=14)
            pdf.savefig(fig_info)
            plt.close(fig_info)

            # Generate distribution graphs based on the chosen metric.
            if args.metric in ["hamming", "both"]:
                # For Hamming, pad sequences so that all have equal length.
                data_hamming = pad_sequences([dict(row) for row in data])
                distribution_hamming = compute_distance_distribution(data_hamming, metric="hamming")
                fig_hamming = create_distance_distribution_figure(distribution_hamming, base, "hamming", filepath)
                pdf.savefig(fig_hamming)
                plt.close(fig_hamming)

            if args.metric in ["levenshtein", "both"]:
                # For Levenshtein, use the original data.
                distribution_lev = compute_distance_distribution(data, metric="levenshtein")
                fig_lev = create_distance_distribution_figure(distribution_lev, base, "levenshtein", filepath)
                pdf.savefig(fig_lev)
                plt.close(fig_lev)

        # Aggregated box plot for N0 values (across all files) using file numbers as labels.
        plt.figure()
        labels = [str(file_number) for file_number, _ in file_n0_data]
        n0_data = [values for _, values in file_n0_data]
        plt.boxplot(n0_data, labels=labels)
        plt.title("Box Plot of N0 Values per File (File Numbers)")
        plt.xlabel("File Number")
        plt.ylabel("N0")
        plt.grid(True)
        pdf.savefig()
        plt.close()

    print(f"All plots saved in '{aggregated_pdf}'.")

if __name__ == "__main__":
    main()
