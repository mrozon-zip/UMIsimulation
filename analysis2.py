#!/usr/bin/env python3
import os
import csv
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D


# ----------------------------
# Helper functions
# ----------------------------
def load_true_barcodes(true_barcodes_file):
    """
    Load true barcode sequences from the given CSV file.
    Expects a header row: "sequence,N0" and then rows with a string in the first column.
    Returns a set of sequences.
    """
    true_seqs = set()
    with open(true_barcodes_file, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        for row in reader:
            if row:
                true_seqs.add(row[0])
    return true_seqs


def parse_parameters_from_filename(filename):
    """
    Parse parameters from filename regardless of order.

    For files that do NOT start with "pcr" (case insensitive), detect:
      - mut_<value>   -> mutation_rate
      - Sr_<value>    -> s_radius
      - dens_<value>  -> density

    For files that start with "pcr" (case insensitive), detect:
      - mut_<value>      -> mutation_rate
      - cycles_<value>   -> cycles

    Returns a tuple:
      For non-pcr files: (mutation_rate, s_radius, density, None)
      For pcr files: (mutation_rate, None, None, cycles)
    """
    filename_lower = filename.lower()
    if filename_lower.startswith("pcr"):
        mut_match = re.search(r"mut_([\d\.]+)", filename)
        cycles_match = re.search(r"cycles_([\d\.]+)", filename)
        mutation_rate = float(mut_match.group(1)) if mut_match else None
        cycles = float(cycles_match.group(1)) if cycles_match else None
        return mutation_rate, None, None, cycles
    else:
        mut_match = re.search(r"mut_([\d\.]+)", filename)
        sr_match = re.search(r"Sr_([\d\.]+)", filename)
        dens_match = re.search(r"dens_([\d\.]+)", filename)
        mutation_rate = float(mut_match.group(1)) if mut_match else None
        s_radius = float(sr_match.group(1)) if sr_match else None
        density = float(dens_match.group(1)) if dens_match else None
        return mutation_rate, s_radius, density, None


def process_file(file_path, true_seqs):
    """
    Processes a given CSV file in the results folder.
    For every row (except header):
       - If the sequence is in true_seqs, add its N0 to true_count.
       - Else, add its N0 to false_count.
    Returns (true_count, false_count).
    """
    true_count = 0
    false_count = 0
    with open(file_path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        for row in reader:
            if row:
                seq = row[0]
                try:
                    n0 = float(row[1])
                except ValueError:
                    n0 = 0
                if seq in true_seqs:
                    true_count += n0
                else:
                    false_count += n0
    return true_count, false_count


# ----------------------------
# Main processing and plotting
# ----------------------------
def main():
    true_barcodes_file = "true_barcodes.csv"
    results_folder = "results1"
    summary_csv = "barcode_counts_summary.csv"
    pdf_filename = "barcode_plots.pdf"

    # Load true barcode sequences.
    true_seqs = load_true_barcodes(true_barcodes_file)

    # Process files: store dictionaries with details for each file.
    # Separate PCR files from non-PCR files.
    non_pcr_files_data = []
    pcr_files_data = []

    for filename in os.listdir(results_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(results_folder, filename)
            mutation_rate, s_radius, density, cycles = parse_parameters_from_filename(filename)
            if mutation_rate is None:
                print(f"Could not parse mutation rate from filename: {filename}")
                continue
            true_count, false_count = process_file(file_path, true_seqs)
            entry = {
                "filename": filename,
                "mutation_rate": mutation_rate,
                "true_count": true_count,
                "false_count": false_count
            }
            if filename.lower().startswith("pcr"):
                entry["cycles"] = cycles
                pcr_files_data.append(entry)
                print(f"{filename} (PCR) -> True count: {true_count}, False count: {false_count}")
            else:
                entry["s_radius"] = s_radius
                entry["density"] = density
                non_pcr_files_data.append(entry)
                print(f"{filename} -> True count: {true_count}, False count: {false_count}")

    # Save summary CSV with filename, true_count, and false_count.
    with open(summary_csv, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["filename", "true_count", "false_count"])
        for entry in non_pcr_files_data + pcr_files_data:
            writer.writerow([entry["filename"], entry["true_count"], entry["false_count"]])
    print(f"\nSummary saved to {summary_csv}")

    with PdfPages(pdf_filename) as pdf:
        # ----------------------------
        # Plotting for non-PCR files (using three x-axis parameters)
        # ----------------------------
        if non_pcr_files_data:
            # Define plotting setups for each x-axis parameter.
            # For each plot the "other parameters" are the two that are not on the x-axis.
            plot_setups = {
                "mutation_rate": {
                    "x_label": "Mutation Rate",
                    "other_keys": ("s_radius", "density"),
                    "legend_label": lambda d: f"Sr: {d['s_radius']}, dens: {d['density']}"
                },
                "s_radius": {
                    "x_label": "S_radius",
                    "other_keys": ("mutation_rate", "density"),
                    "legend_label": lambda d: f"mut: {d['mutation_rate']}, dens: {d['density']}"
                },
                "density": {
                    "x_label": "Density",
                    "other_keys": ("mutation_rate", "s_radius"),
                    "legend_label": lambda d: f"mut: {d['mutation_rate']}, Sr: {d['s_radius']}"
                }
            }
            # For each x-axis parameter, create one plot for true counts and one for false counts.
            for x_param, setup in plot_setups.items():
                for count_type in ["true_count", "false_count"]:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    other_keys = setup["other_keys"]
                    # Determine unique combinations of the "other parameters" for color mapping.
                    unique_keys = {}
                    cmap = plt.get_cmap('tab20')
                    color_idx = 0
                    for data in non_pcr_files_data:
                        key = (data[other_keys[0]], data[other_keys[1]])
                        if key not in unique_keys:
                            unique_keys[key] = cmap(color_idx % 20)
                            color_idx += 1

                    # Create a dictionary to store legend handles for each unique key.
                    legend_handles = {}

                    # Plot each file's single count (true or false).
                    for data in non_pcr_files_data:
                        x_val = data[x_param]
                        y_val = data[count_type]
                        key = (data[other_keys[0]], data[other_keys[1]])
                        color = unique_keys[key]
                        ax.scatter(x_val, y_val, color=color, marker='o', s=80)
                        label = setup["legend_label"](data)
                        if key not in legend_handles:
                            legend_handles[key] = Line2D([0], [0], marker='o', color=color, linestyle='',
                                                         markersize=8, label=label)
                    ax.set_xlabel(setup["x_label"])
                    if count_type == "true_count":
                        ax.set_ylabel("True Count")
                        title_type = "True Barcode Counts"
                    else:
                        ax.set_ylabel("False Count")
                        title_type = "False Barcode Counts"
                    ax.set_title(f"{title_type} vs. {setup['x_label']}")
                    ax.legend(handles=list(legend_handles.values()), title="Other Parameters", loc='best',
                              fontsize='small')
                    ax.grid(True, linestyle='--', alpha=0.5)
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
        else:
            print("No non-PCR files found for plotting.")

        # ----------------------------
        # Plotting for PCR files
        # ----------------------------
        if pcr_files_data:
            # For PCR files, create two plots: one for true barcode counts and one for false barcode counts.
            # X axis: mutation_rate; color determined by cycles.
            for count_type in ["true_count", "false_count"]:
                fig, ax = plt.subplots(figsize=(8, 6))
                unique_cycles = {}
                cmap = plt.get_cmap('tab10')
                color_idx = 0
                legend_handles = {}

                for data in pcr_files_data:
                    x_val = data["mutation_rate"]
                    y_val = data[count_type]
                    cycle = data.get("cycles", "N/A")
                    if cycle not in unique_cycles:
                        unique_cycles[cycle] = cmap(color_idx % 10)
                        color_idx += 1
                    color = unique_cycles[cycle]
                    ax.scatter(x_val, y_val, color=color, marker='o', s=80)
                    if cycle not in legend_handles:
                        legend_handles[cycle] = Line2D([0], [0], marker='o', color=color, linestyle='',
                                                       markersize=8, label=f"Cycles: {cycle}")
                ax.set_xlabel("Mutation Rate")
                if count_type == "true_count":
                    ax.set_ylabel("True Count")
                    title_type = "True Barcode Counts (PCR)"
                else:
                    ax.set_ylabel("False Count")
                    title_type = "False Barcode Counts (PCR)"
                ax.set_title(f"{title_type} vs. Mutation Rate")
                ax.legend(handles=list(legend_handles.values()), title="Cycles", loc='best', fontsize='small')
                ax.grid(True, linestyle='--', alpha=0.5)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
        else:
            print("No PCR files found for plotting.")

    print(f"\nPlots saved to {pdf_filename}")


if __name__ == "__main__":
    main()
