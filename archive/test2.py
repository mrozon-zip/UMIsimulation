import subprocess
import os
import sys

def run_command(cmd):
    """Run a command and raise an error if it fails."""
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

AMPLIFIED_DIR = "../results_amplified"
DENOISED_DIR   = "../results_denoised"
TRUE_BARCODES  = "true_barcodes.csv"
analysis_type = 2

# Collect and sort all CSVs
amp_files = sorted(f for f in os.listdir(AMPLIFIED_DIR) if f.endswith('.csv'))
if analysis_type == 2:
    den_files = sorted(f for f in os.listdir(DENOISED_DIR) if 'type2' in f and f.endswith('.csv'))
elif analysis_type == 1:
    den_files = sorted(f for f in os.listdir(DENOISED_DIR) if 'type2' not in f and f.endswith('.csv'))

# Ensure they match one-to-one by basename
amp_bases = [os.path.splitext(f)[0] for f in amp_files]
den_bases = [os.path.splitext(f)[0] for f in den_files]
if len(amp_bases) != len(den_bases):
    sys.exit(f"Error: mismatch between amplified and denoised files:\n  {len(amp_bases)}{amp_bases}\n  {len(den_bases)}{den_bases}")

# Loop and call analyse for each
for base in amp_bases:
    amp_path = os.path.join(AMPLIFIED_DIR, f"{base}.csv")
    if analysis_type == 1:
        den_path = os.path.join(DENOISED_DIR,   f"{base}_denoised.csv")
    elif analysis_type == 2:
        den_path = os.path.join(DENOISED_DIR, f"{base}_type2_denoised.csv")
    cmd = [
        "python3", "main.py", "analyse",
        "--denoised",      den_path,
        "--amplified",     amp_path,
        "--true_barcodes", TRUE_BARCODES,
    ]
    print("Running:", " ".join(cmd))
    run_command(cmd)