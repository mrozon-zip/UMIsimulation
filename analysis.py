import csv
import matplotlib.pyplot as plt
import seaborn as sns
import os


def confusion_matrix(denoised_file, amplified, true_umis_file):
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

    cm = [[tp, fp], [fn, tn]]

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Actual Positives", "Actual Negatives"],
                yticklabels=["Predicted Positives", "Predicted Negatives"])
    plt.title("Confusion Matrix")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")

    # Save the plot to a png file in the results1 folder with the same name as the amplified file
    os.makedirs("results1", exist_ok=True)
    base_name = os.path.basename(amplified)
    plot_file = os.path.join("results1", os.path.splitext(base_name)[0] + ".png")
    plt.savefig(plot_file)

    plt.show()

    print(f"True Positives (TP): {tp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")

    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}
