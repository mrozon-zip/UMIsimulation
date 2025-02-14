import seaborn as sns
from classes import Denoiser, SimPcr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import time
from itertools import product
import csv

# Parameters to iterate over
pcr_cycles = [10]
barcode_lengths = [12]
pcr_efficiencies = [0.85]
error_probabilities = [0.0001]
initial_barcodes = [200]

# Choose denoiser method:
method = "directional"  # type "directional" or "simple"

# File paths for the data
true_umis_file = "true_UMIs.csv"
#
# Output PDF for results
output_pdf = "hyperparameter_results_with_metrics1.pdf"

# Variables to track the best and worst runs
best_run = {"precision": 0, "accuracy": 0, "text": ""}
worst_run = {"precision": float("inf"), "accuracy": float("inf"), "text": ""}
most_precise_run = {"precision": 0, "text": ""}
most_accurate_run = {"accuracy": 0, "text": ""}

# Create a PDF to store the results
with PdfPages(output_pdf) as pdf:
    # Use itertools.product to create the parameter grid
    parameter_grid = product(pcr_cycles, barcode_lengths, pcr_efficiencies, error_probabilities, initial_barcodes)

    times = []
    unique_sequences = []

    for cycles, length, efficiency, error_prob, num_barcodes in parameter_grid:
        # Skip higher initial barcode values except for minimal parameters
        if num_barcodes > 600 and (cycles != min(pcr_cycles) or length != min(barcode_lengths) or
                                   efficiency != max(pcr_efficiencies) or
                                   error_prob != min(error_probabilities)):
            continue

        print(f"Running with parameters: "
              f"Cycles={cycles}, Length={length}, Efficiency={efficiency}, "
              f"Error Probability={error_prob}, Initial Barcodes={num_barcodes}")

        start_time = time.time()

        # Run simulation
        simulator = SimPcr(length=length, number_of_rows=num_barcodes)
        simulator.create_true_umis()

        error_types = {
            'substitution': 0.6,
            'deletion': 0.2,
            'insertion': 0.2
        }
        simulator.amplify_with_errors(
            amplification_probability=efficiency,
            error_rate=error_prob,
            error_types=error_types,
            amplification_cycles=cycles
        )

        with open("amplified_UMIs.csv", mode='r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if len(rows) > 15000000:
                continue

        # Perform denoising
        denoiser = Denoiser(csv_file='amplified_UMIs.csv')
        if method == "directional":
            before_graph, after_graph, unique_molecules = denoiser.directional_networks(show=0)
            denoiser.networks_resolver(after_graph, toggle="central_node")
            denoised_file = "directional_results.csv"
        elif method == "simple":
            threshold_value = 300
            denoiser.simple(threshold_value)
            denoised_file = "simple_result.csv"
        else:
            print("Invalid input in method choice")
            break

        results = denoiser.analysis(denoised_file, true_umis_file)
        end_time = time.time()
        runtime = end_time - start_time

        tp = results["TP"]
        tn = results["TN"]
        fp = results["FP"]
        fn = results["FN"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

        cm = [[tp, fp], [fn, tn]]

        # Save the confusion matrix heatmap to the PDF
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Actual Positives", "Actual Negatives"],
                    yticklabels=["Predicted Positives", "Predicted Negatives"], ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        pdf.savefig(fig)
        plt.close()

        # Save textual results to the PDF
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        text = (
            f"Parameter Set:\n"
            f"- PCR Cycles: {cycles}\n"
            f"- Barcode Length: {length}\n"
            f"- PCR Efficiency: {efficiency}\n"
            f"- Error Probability: {error_prob}\n"
            f"- Initial Barcodes: {num_barcodes}\n\n"
            f"- True Positives (TP): {tp}\n"
            f"- True Negatives (TN): {tn}\n"
            f"- False Positives (FP): {fp}\n"
            f"- False Negatives (FN): {fn}\n"
            f"- Precision: {precision:.4f}\n"
            f"- Accuracy: {accuracy:.4f}\n"
            f"- Runtime: {runtime:.2f} seconds\n"
        )
        ax.text(0.5, 0.5, text, fontsize=12, va='center', ha='center', wrap=True)
        pdf.savefig(fig)
        plt.close()

    # Add summary page
    average_ratio = sum(unique_sequences) / sum(times) if sum(times) > 0 else 0
    summary_fig, summary_ax = plt.subplots(figsize=(8, 6))
    summary_ax.axis('off')
    summary_text = (
        f"Summary of Results:\n\n"
        f"- Best Run: {best_run['text']} (Precision={best_run['precision']:.4f}, Accuracy={best_run['accuracy']:.4f})\n"
        f"- Worst Run: {worst_run['text']} (Precision={worst_run['precision']:.4f}, Accuracy={worst_run['accuracy']:.4f})\n"
        f"- Most Precise Run: {most_precise_run['text']} (Precision={most_precise_run['precision']:.4f})\n"
        f"- Most Accurate Run: {most_accurate_run['text']} (Accuracy={most_accurate_run['accuracy']:.4f})\n\n"
        f"- Average Time Per Molecule: {average_ratio:.4f} seconds\n"
    )
    summary_ax.text(0.5, 0.5, summary_text, fontsize=12, va='center', ha='center', wrap=True)
    pdf.savefig(summary_fig)
    plt.close()

print(f"All results saved to {output_pdf}.")
