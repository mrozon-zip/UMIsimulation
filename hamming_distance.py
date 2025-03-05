#!/usr/bin/env python3
import csv
import os
import argparse
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


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
        raise ValueError("Insufficient data to meet the required cumulative N0 and minimum row count.")

    return sampled_data
        #if len(rows) >= 50_000:
        #    # Use the provided sample_size if dataset is large.
        #    sampled_data = random.sample(rows, sample_size)
        #    print("Dataset size bigger than 50_000, sample might be lower than 20%")
        #else:
        #    k = int(0.2 * len(rows))
        #    sampled_data = random.sample(rows, k)
        #    print("Sampling 20% of the dataset")


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

    For each pair of rows (i, j) (with i < j), the weight is the product of the N0 values
    of the two sequences. That is, if one row's count is A and the other's is B, then the
    computed distance is recorded A * B times.

    Returns a tuple (distribution, weighted_distances) where:
      - distribution: dict mapping distance -> weighted count,
      - weighted_distances: list of computed distance values repeated according to the weight.
    """
    weighted_distances = []
    n = len(sequences)
    for i in range(n):
        for j in range(i + 1, n):
            if metric == "hamming":
                d = hamming_distance(sequences[i]['sequence'], sequences[j]['sequence'])
            elif metric == "levenshtein":
                d = levenshtein_distance(sequences[i]['sequence'], sequences[j]['sequence'])
            # Multiply the N0 values for the two sequences to determine the weight
            weight = int(sequences[i]['N0']) * int(sequences[j]['N0'])
            weighted_distances.extend([d] * weight)
    distribution = {}
    for d in weighted_distances:
        distribution[d] = distribution.get(d, 0) + 1
    return distribution, weighted_distances


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


def main():
    parser = argparse.ArgumentParser(
        description="Compute distance distributions and produce plots for CSV files."
    )
    parser.add_argument("csv_files", nargs="+", help="List of CSV files to process")
    parser.add_argument("--metric", type=str, choices=["hamming", "levenshtein", "both"],
                        default="hamming",
                        help="Distance metric to use: 'hamming' (requires equal length), 'levenshtein', or 'both'.")
    parser.add_argument("--sample_size", type=int, default=1000, help="Sample size for analysis")
    parser.add_argument("--minrows", type=int, default=100, help="Sample size for analysis")
    args = parser.parse_args()

    # Dictionaries for aggregated data
    n0_values_per_file = {}                # {filepath: [N0, ...]}
    hamming_distance_values_per_file = {}  # Only populated if metric in ("hamming", "both")
    levenshtein_distance_values_per_file = {}  # Only populated if metric in ("levenshtein", "both")

    # Create one aggregated PDF file for all plots.
    aggregated_pdf = "all_plots.pdf"
    with PdfPages(aggregated_pdf) as pdf:
        # Process each CSV file
        for filepath in args.csv_files:
            print(f"Processing {filepath}...")
            data = read_csv_file(filepath, args.sample_size, args.minrows)
            base, _ = os.path.splitext(filepath)
            n0_values = [int(row['N0']) for row in data]
            n0_values_per_file[filepath] = n0_values

            if args.metric in ["hamming", "both"]:
                # For Hamming, pad sequences so that all have equal length.
                data_hamming = pad_sequences([dict(row) for row in data])
                # Compute weighted distances using the modified function.
                distribution_hamming, distances_hamming = compute_distance_distribution(data_hamming, metric="hamming")
                hamming_distance_values_per_file[filepath] = distances_hamming
                fig_hamming = create_distance_distribution_figure(distribution_hamming, base, "hamming", filepath)
                pdf.savefig(fig_hamming)
                plt.close(fig_hamming)

            if args.metric in ["levenshtein", "both"]:
                # For Levenshtein, use the original data (with the N0 values).
                distribution_lev, distances_lev = compute_distance_distribution(data, metric="levenshtein")
                levenshtein_distance_values_per_file[filepath] = distances_lev
                fig_lev = create_distance_distribution_figure(distribution_lev, base, "levenshtein", filepath)
                pdf.savefig(fig_lev)
                plt.close(fig_lev)

        # Aggregated box plot for N0 values (across all files)
        plt.figure()
        labels = []
        n0_data = []
        for f, values in n0_values_per_file.items():
            labels.append(os.path.basename(f))
            n0_data.append(values)
        plt.boxplot(n0_data, labels=labels)
        plt.title("Box Plot of N0 Values per File")
        plt.xlabel("File")
        plt.ylabel("N0")
        plt.grid(True)
        pdf.savefig()
        plt.close()

        # Aggregated box plots for distance values (if applicable)
        if args.metric in ["hamming", "both"]:
            plt.figure()
            labels = []
            ham_data = []
            for f, distances in hamming_distance_values_per_file.items():
                labels.append(os.path.basename(f))
                ham_data.append(distances)
            plt.boxplot(ham_data, labels=labels)
            plt.title("Box Plot of Hamming Distances per File")
            plt.xlabel("File")
            plt.ylabel("Hamming Distance")
            plt.grid(True)
            pdf.savefig()
            plt.close()

        if args.metric in ["levenshtein", "both"]:
            plt.figure()
            labels = []
            lev_data = []
            for f, distances in levenshtein_distance_values_per_file.items():
                labels.append(os.path.basename(f))
                lev_data.append(distances)
            plt.boxplot(lev_data, labels=labels)
            plt.title("Box Plot of Levenshtein Distances per File")
            plt.xlabel("File")
            plt.ylabel("Levenshtein Distance")
            plt.grid(True)
            pdf.savefig()
            plt.close()

    print(f"All plots saved in '{aggregated_pdf}'.")


if __name__ == "__main__":
    main()
