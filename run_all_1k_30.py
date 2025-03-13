import concurrent.futures
import subprocess
import glob

def run_command(cmd):
    """Run a command and raise an error if it fails."""
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

# Collect all commands in a list.
commands = []

## PCR method: cycles and mutation_rate parameters.
#for cycles in [15, 25, 30]:
#    for mutation_rate in [0, 0.002, 0.005, 0.01]:
#        cmd = [
#            "python3", "main.py", "amplify",
#            "--method", "pcr",
#            "--mutation_rate", str(mutation_rate),
#            "--cycles", str(cycles),
#            "--output", f"mut_{mutation_rate}_{cycles}.csv",
#            "--no_plot"
#        ]
#        commands.append(cmd)

# Bridge method: parameters S_radius, density, success_prob, deviation, and mutation_rate.
#for s_radius in [5, 7.5, 10]:
#    for density in [5, 7.5]:
#        for success_prob in [0.85]:
#            for deviation in [0.05]:
#                for mutation_rate in [0, 0.001, 0.002,]:
#                    cmd = [
#                        "python3", "main.py", "amplify",
#                        "--method", "bridge",
#                        "--mutation_rate", str(mutation_rate),
#                        "--S_radius", str(s_radius),
#                        "--density", str(density),
#                        "--success_prob", str(success_prob),
#                        "--deviation", str(deviation),
#                        "--simulate",
#                        "--no_plot",
#                        "--output", f"mut_{mutation_rate}_Sr{s_radius}_dens{density}_SP_{success_prob}_dev{deviation}.csv"
#                    ]
#                    commands.append(cmd)

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

# After all amplification commands are finished, run hamming_distance.py on the results.
csv_files = glob.glob("results1/*.csv")
hamming_cmd = ["python3", "hamming_distance.py"] + csv_files + ["--metric", "both"]
run_command(hamming_cmd)