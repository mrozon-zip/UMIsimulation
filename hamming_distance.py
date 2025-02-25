#!/usr/bin/env python3
import csv
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def read_csv_file(filepath):
    """Read a CSV file and return a list of dictionaries."""
    with open(filepath, 'r', newline='') as f:
        reader = csv.DictReader(f)
        return list(reader)


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
    Compute the distribution of pairwise distances (Hamming or Levenshtein)
    using only unique comparisons.
    Returns a tuple (distribution, distances) where:
      distribution: dict mapping distance -> count,
      distances: list of computed distance values.
    """
    distances = []
    n = len(sequences)
    for i in range(n):
        for j in range(i + 1, n):
            if metric == "hamming":
                d = hamming_distance(sequences[i], sequences[j])
            elif metric == "levenshtein":
                d = levenshtein_distance(sequences[i], sequences[j])
            distances.append(d)
    distribution = {}
    for d in distances:
        distribution[d] = distribution.get(d, 0) + 1
    return distribution, distances


def create_distance_distribution_figure(distribution, base, metric):
    """
    Create a bar chart figure for the distance distribution.
    The y-axis shows percentage values.
    """
    fig = plt.figure()
    dists = sorted(distribution.keys())
    counts = [distribution[d] for d in dists]
    total = sum(counts)
    percentages = [(count / total) * 100 for count in counts]

    plt.bar(dists, percentages, width=0.8, color='skyblue', edgecolor='black')
    plt.xlabel(f"{metric.capitalize()} Distance")
    plt.ylabel("Percentage (%)")
    plt.title(f"{metric.capitalize()} Distance Distribution for {base}")
    plt.grid(True)
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Compute distance distributions and produce plots for CSV files."
    )
    parser.add_argument("csv_files", nargs="+", help="List of CSV files to process")
    parser.add_argument("--metric", type=str, choices=["hamming", "levenshtein", "both"],
                        default="hamming",
                        help="Distance metric to use: 'hamming' (requires equal length), 'levenshtein', or 'both'.")
    args = parser.parse_args()

    # Dictionaries for aggregated data
    n0_values_per_file = {}  # {filepath: [N0, ...]}
    hamming_distance_values_per_file = {}  # Only populated if metric in ("hamming", "both")
    levenshtein_distance_values_per_file = {}  # Only populated if metric in ("levenshtein", "both")

    # Process each file individually
    for filepath in args.csv_files:
        print(f"Processing {filepath}...")
        data = read_csv_file(filepath)
        base, _ = os.path.splitext(filepath)
        n0_values = [int(row['N0']) for row in data]
        n0_values_per_file[filepath] = n0_values

        # List to hold figures for the per-file PDF
        figures = []

        if args.metric in ["hamming", "both"]:
            # For Hamming, pad sequences so all have equal length.
            data_hamming = pad_sequences([dict(row) for row in data])
            sequences_hamming = [row['sequence'] for row in data_hamming]
            distribution_hamming, distances_hamming = compute_distance_distribution(sequences_hamming, metric="hamming")
            hamming_distance_values_per_file[filepath] = distances_hamming
            fig_hamming = create_distance_distribution_figure(distribution_hamming, base, "hamming")
            figures.append(fig_hamming)

        if args.metric in ["levenshtein", "both"]:
            # For Levenshtein, use the original sequences (no padding needed).
            sequences_lev = [row['sequence'] for row in data]
            distribution_lev, distances_lev = compute_distance_distribution(sequences_lev, metric="levenshtein")
            levenshtein_distance_values_per_file[filepath] = distances_lev
            fig_lev = create_distance_distribution_figure(distribution_lev, base, "levenshtein")
            figures.append(fig_lev)

        # Save all figures for this file into one PDF.
        if args.metric == "both":
            pdf_filename = f"{base}_plots.pdf"
        else:
            pdf_filename = f"{base}_{args.metric}_plots.pdf"
        with PdfPages(pdf_filename) as pdf:
            for fig in figures:
                pdf.savefig(fig)
                plt.close(fig)
        print(f"Saved plots for {filepath} in '{pdf_filename}'")

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
    n0_box_pdf = "n0_boxplot.pdf"
    plt.savefig(n0_box_pdf)
    print(f"N0 box plot saved as '{n0_box_pdf}'")
    plt.show()

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
        ham_box_pdf = "hamming_boxplot.pdf"
        plt.savefig(ham_box_pdf)
        print(f"Hamming box plot saved as '{ham_box_pdf}'")
        plt.show()

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
        lev_box_pdf = "levenshtein_boxplot.pdf"
        plt.savefig(lev_box_pdf)
        print(f"Levenshtein box plot saved as '{lev_box_pdf}'")
        plt.show()


if __name__ == "__main__":
    main()