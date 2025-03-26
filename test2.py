import subprocess
import glob

def run_command(cmd):
    """Run a command and raise an error if it fails."""
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

# After all amplification commands are finished, run hamming_distance.py on the results.
csv_files = glob.glob("results1/*.csv")
hamming_cmd = ["python3", "hamming_distance.py"] + csv_files + ["--metric", "both"]
run_command(hamming_cmd)