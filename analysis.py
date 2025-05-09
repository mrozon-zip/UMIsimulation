import csv
import matplotlib.pyplot as plt
import seaborn as sns
import os

def confusion_matrix(denoised_file, amplified, true_umis_file, plot):
    def load_file(file):
        with open(file, mode='r') as f:
            return {row['sequence'] for row in csv.DictReader(f)}

    sequences_denoised = load_file(denoised_file)
    sequences_true = load_file(true_umis_file)
    sequences_amplified = load_file(amplified)

    tp = len(sequences_denoised & sequences_true)
    fn = len(sequences_true - sequences_denoised)
    fp = len(sequences_denoised - sequences_true)
    tn = len(sequences_amplified - sequences_denoised - sequences_true)

    # Calculate metrics
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total != 0 else 0
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

    if plot:
        cm = [[tp, fp], [fn, tn]]

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Actual Positives", "Actual Negatives"],
                    yticklabels=["Predicted Positives", "Predicted Negatives"])
        plt.title("Confusion Matrix")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")

        # Save the plot to a png file in the results1 folder with the same name as the amplified file
        os.makedirs("dump/results1", exist_ok=True)
        base_name = os.path.basename(amplified)
        plot_file = os.path.join("dump/results1", os.path.splitext(base_name)[0] + ".png")
        plt.savefig(plot_file)

        plt.show()

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity
    }

