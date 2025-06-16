import concurrent.futures
import subprocess
import glob
import os

def run_command(cmd):
    """Run a command and raise an error if it fails."""
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

# Collect all commands in a list.
commands = []


for s_radius in [30]:
    for density in [150]:
        for AOE in [10, 50]:
            for mutation_rate in [0, 0.001, 0.002, 0.01]:
                cmd = [
                    "python3", "main.py", "amplify",
                    "--method", "polonies_amplification",
                    "--mutation_rate", str(mutation_rate),
                    "--S_radius", str(s_radius),
                    "--density", str(density),
                    "--AOE_radius", str(AOE),
                    "--no_simulate",
                    "--no_plot",
                    "--output", f"mut_{mutation_rate}_Sr_{s_radius}_dens_{density}_AOE_{AOE}.csv"
                ]
                commands.append(cmd)

max_workers = 4
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