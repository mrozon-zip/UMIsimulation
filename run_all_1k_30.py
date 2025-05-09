import concurrent.futures
import subprocess
import glob
import os
import sys

def run_command(cmd):
    """Run a command and raise an error if it fails."""
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

# Collect all commands in a list.
commands = []

# PCR method: cycles and mutation_rate parameters.
for cycles in [15, 25, 30]: #
    for mutation_rate in [0, 0.002, 0.005, 0.01]:
        cmd = [
            "python3", "main.py", "amplify",
            "--method", "pcr",
            "--mutation_rate", str(mutation_rate),
            "--cycles", str(cycles),
            "--output", f"mut_{mutation_rate}_{cycles}.csv",
            "--no_plot"
        ]
        commands.append(cmd)

# Polonies Amplification method: similar to bridge, but with different S_radius values.
for s_radius in [10, 20, 30]:
    for density in [50, 100, 150]:
        for success_prob in [0.85]:
            for deviation in [0.05]:
                for mutation_rate in [0, 0.001, 0.002, 0.01]:
                    cmd = [
                        "python3", "main.py", "amplify",
                        "--method", "polonies_amplification",
                        "--mutation_rate", str(mutation_rate),
                        "--S_radius", str(s_radius),
                        "--density", str(density),
                        "--success_prob", str(success_prob),
                        "--deviation", str(deviation),
                        "--no_simulate",
                        "--no_plot",
                        "--output", f"mut_{mutation_rate}_Sr_{s_radius}_dens_{density}_SP_{success_prob}_dev_{deviation}.csv"
                    ]
                    commands.append(cmd)

# Run all commands in parallel using a ThreadPoolExecutor.
# (Choose max_workers based on your system's capacity; here we use 8 as an example.)
max_workers = 1
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(run_command, cmd): cmd for cmd in commands}
    for future in concurrent.futures.as_completed(futures):
        cmd = futures[future]
        try:
            future.result()
        except Exception as exc:
            print(f"Command {' '.join(cmd)} generated an exception: {exc}")
        else:
            print(f"Command {' '.join(cmd)} completed successfully.")

# find all CSVs in results_amplified/
csv_files = glob.glob(os.path.join("results_amplified", "*.csv"))
# run denoise for each one
for csv_path in csv_files:
    cmd = [
        "python3", "main.py", "denoise",
        "--input", csv_path,
        "--type", "1"
    ]
    run_command(cmd)

AMPLIFIED_DIR = "results_amplified"
DENOISED_DIR   = "results_denoised"
TRUE_BARCODES  = "true_barcodes.csv"

# Collect and sort all CSVs
amp_files = sorted(f for f in os.listdir(AMPLIFIED_DIR) if f.endswith('.csv'))
den_files = sorted(f for f in os.listdir(DENOISED_DIR)   if f.endswith('.csv'))

# Ensure they match one-to-one by basename
amp_bases = [os.path.splitext(f)[0] for f in amp_files]
den_bases = [os.path.splitext(f)[0] for f in den_files]
if amp_bases != den_bases:
    sys.exit(f"Error: mismatch between amplified and denoised files:\n  {amp_bases}\n  {den_bases}")

# Loop and call analyse for each
for base in amp_bases:
    amp_path = os.path.join(AMPLIFIED_DIR, f"{base}.csv")
    den_path = os.path.join(DENOISED_DIR,   f"{base}.csv")
    cmd = [
        "python3", "main.py", "analyse",
        "--denoised",      den_path,
        "--amplified",     amp_path,
        "--true_barcodes", TRUE_BARCODES,
        "--plot",          "False"
    ]
    print("Running:", " ".join(cmd))
    run_command(cmd)

# After all amplification commands are finished, run hamming_distance.py on the results.
#csv_files = glob.glob("results_amplified/*.csv")
#hamming_cmd = ["python3", "hamming_distance.py"] + csv_files + ["--metric", "both"]
#run_command(hamming_cmd)