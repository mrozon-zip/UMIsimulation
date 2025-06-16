#!/usr/bin/env python3

"""
This script generates an amplification curve based on provided file
"""

import os
import glob
import pandas as pd
import ast
import matplotlib.pyplot as plt
import re
import scipy as sp
import numpy as np
from matplotlib.ticker import MultipleLocator


def get_born_values_from_file(filepath):
    # Read CSV and flatten all "born" values (which may be scalars or lists)
    df = pd.read_csv(filepath)
    born_values = []
    for val in df['born']:
        if pd.isnull(val):
            continue
        try:
            parsed = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            parsed = val
        if isinstance(parsed, (list, tuple)):
            candidates = parsed
        else:
            s = str(parsed).strip()
            if s.startswith('[') and s.endswith(']'):
                s = s[1:-1]
            for sep in [',', ';']:
                if sep in s:
                    candidates = s.split(sep)
                    break
            else:
                candidates = [s]
        for item in candidates:
            try:
                born_values.append(int(item))
            except (ValueError, TypeError):
                continue
    return born_values


def plot_amplification_curve(born_values, label, color):
    if not born_values:
        print(f"No born values for {label}, skipping.")
        return
    max_cycle = max(born_values)
    cycles = list(range(1, max_cycle + 1))
    counts = [sum(1 for v in born_values if v <= n) for n in cycles]
    plt.plot(cycles, counts, marker='o', linestyle='-', label=label, color=color)


def sigmoid(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))


# New function to report files with lowest and highest maximum born values
def report_extreme_files(files):
    """
    Print the names of the files with the lowest and highest maximum 'born' values.
    """
    max_values = {}
    for filepath in files:
        born_values = get_born_values_from_file(filepath)
        if born_values:
            max_values[filepath] = max(born_values)
    if not max_values:
        print("No born values found in any files.")
        return
    min_file = min(max_values, key=max_values.get)
    max_file = max(max_values, key=max_values.get)
    print(f"File with lowest maximum born value: {os.path.basename(min_file)} ({max_values[min_file]})")
    print(f"File with highest maximum born value: {os.path.basename(max_file)} ({max_values[max_file]})")


def fit_sigmoid_to_file(filepath):
    born_values = get_born_values_from_file(filepath)
    if not born_values:
        return None
    max_cycle = max(born_values)
    x = np.arange(1, max_cycle + 1)
    y = np.array([sum(1 for v in born_values if v <= n) for n in x])

    # Estimate approximate starting values for parameters
    L_guess = float(y.max())
    k_guess = 1.0
    x0_guess = float(np.median(x))
    p0 = [L_guess, k_guess, x0_guess]
    try:
        popt, _ = sp.optimize.curve_fit(sigmoid, x, y, p0, maxfev=10000)
    except RuntimeError:
        return None
    L, k, x0 = [round(param, 3) for param in popt]
    equation = f"y = {L} / (1 + exp(-{k} * (x - {x0})))"

    # Calculate predicted values and R^2
    y_pred = sigmoid(x, *popt)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    r2 = round(r2, 4)

    return equation, r2


def fit_sigmoids_and_save(filepaths, out_csv="sigmoid_fits.csv"):
    results = []
    for fp in filepaths:
        res = fit_sigmoid_to_file(fp)
        filename = os.path.basename(fp)
        if res is not None:
            eq, r2 = res
            results.append([filename, eq, r2])
            print(f"{filename}: {eq} | R^2 = {r2}")
        else:
            results.append([filename, "Fit failed", "N/A"])
            print(f"{filename}: Fit failed")
    df = pd.DataFrame(results, columns=["Filename", "Sigmoid Equation", "R^2"])
    df.to_csv(out_csv, index=False)
    print(f"\nSaved results to: {out_csv}")


def parse_sigmoid_equation(eq_str):
    """
    Extracts parameters L, k, x0 from string like 'y = L / (1 + exp(-k * (x - x0)))'
    """
    import re
    eq_str_nospace = eq_str.replace(" ", "")
    pattern = r"y=([\d\.eE+-]+)/\(1\+exp\(-([\d\.eE+-]+)\*\(x-([\d\.eE+-]+)\)\)\)"
    match = re.match(pattern, eq_str_nospace)
    if match:
        L, k, x0 = match.groups()
        return float(L), float(k), float(x0)
    else:
        raise ValueError("Could not parse equation: " + eq_str)


def compare_sigmoid_rows(csv_path):
    # Load file and show numbered lines
    df = pd.read_csv(csv_path)
    print("Select two rows to compare (enter indices):")
    for i, row in df.iterrows():
        print(f"{i}: {row['Filename']} | {row['Sigmoid Equation']} | R² = {row['R^2']}")

    idx1 = int(input("Enter index of the first row: "))
    idx2 = int(input("Enter index of the second row: "))

    eq1 = df.iloc[idx1]['Sigmoid Equation']
    eq2 = df.iloc[idx2]['Sigmoid Equation']

    try:
        params1 = parse_sigmoid_equation(eq1)
        params2 = parse_sigmoid_equation(eq2)
    except Exception as e:
        print(f"Could not parse equations: {e}")
        return

    param_dist = np.linalg.norm(np.array(params1) - np.array(params2))

    x = np.arange(1, 31)
    y1 = sigmoid(x, *params1)
    y2 = sigmoid(x, *params2)
    pearson_corr = np.corrcoef(y1, y2)[0, 1]

    print("\nCurve comparison:")
    print(f"Euclidean distance between parameters (L, k, x0): {param_dist:.4f}")
    print(f"Pearson correlation (y1 vs y2, x=1..30): {pearson_corr:.4f}")


def fit_sigmoids_with_comparisons(filepaths, out_csv="sigmoid_fits.csv"):
    """
    Fit sigmoid for each file, save params. Then compare all fits to a reference file selected by the user.
    """
    fits = []
    for fp in filepaths:
        res = fit_sigmoid_to_file(fp)
        filename = os.path.basename(fp)
        if res is not None:
            eq, r2 = res
            fits.append({'filename': filename, 'filepath': fp, 'eq': eq, 'r2': r2})
        else:
            fits.append({'filename': filename, 'filepath': fp, 'eq': "Fit failed", 'r2': "N/A"})

    print("\n=== List of files ===")
    for i, item in enumerate(fits):
        print(f"{i}: {item['filename']} | {item['eq']} | R² = {item['r2']}")
    idx_ref = int(input("Enter index of the reference file (e.g. 0): "))
    ref = fits[idx_ref]
    try:
        ref_params = parse_sigmoid_equation(ref['eq'])
    except Exception as e:
        print(f"Could not parse reference equation: {e}")
        return

    x = np.arange(1, 31)
    try:
        y_ref = sigmoid(x, *ref_params)
    except Exception:
        y_ref = None

    for item in fits:
        try:
            params = parse_sigmoid_equation(item['eq'])
            if item is ref:
                item['euclid'] = 0.0
            else:
                item['euclid'] = float(np.linalg.norm(np.array(params) - np.array(ref_params)))
        except Exception:
            item['euclid'] = "N/A"

        try:
            y = sigmoid(x, *parse_sigmoid_equation(item['eq']))
            if item is ref:
                item['pearson'] = 1.0
            else:
                if y_ref is not None:
                    item['pearson'] = float(np.corrcoef(y, y_ref)[0, 1])
                else:
                    item['pearson'] = "N/A"
        except Exception:
            item['pearson'] = "N/A"

    df = pd.DataFrame([
        [item['filename'], item['eq'], item['r2'], item['euclid'], item['pearson']]
        for item in fits
    ], columns=["Filename", "Sigmoid Equation", "R^2", "Euclidean Distance", "Pearson Correlation"])
    df.to_csv(out_csv, index=False)
    print(f"\nSaved comparative results to: {out_csv}")


def main():
    files = glob.glob("results_amplified/polonies_mut_0.002_Sr_30_dens_*_AOE*.csv")
    density_groups = {}
    for filepath in files:
        name = os.path.basename(filepath)
        match = re.search(r'dens_(\d+)', name)
        dens = match.group(1) if match else None
        if dens:
            density_groups.setdefault(dens, []).append(filepath)
    density_counters = {dens: 0 for dens in density_groups}
    if not files:
        print("No files matching '*pcr*.csv' or '*polonies*dens_100*.csv' found.")
        return

    plt.figure()
    for filepath in files:
        name = os.path.basename(filepath)
        born = get_born_values_from_file(filepath)
        if 'pcr' in name.lower():
            color = 'red'
            label = 'PCR'
        else:
            match = re.search(r'dens_(\d+)', name)
            dens = match.group(1) if match else 'unknown'
            primers = int(30**2 * 3.14 * int(dens))
            label = f'Polonies - Primers no. {primers}'
            if dens == '100':
                cmap = plt.cm.Blues
            elif dens == '50':
                cmap = plt.cm.Purples
            elif dens == '150':
                cmap = plt.cm.Greens
            else:
                cmap = plt.cm.Greys
            group_len = len(density_groups.get(dens, []))
            idx = density_counters.get(dens, 0)
            shade = (idx + 1) / (group_len + 1) if group_len else 0.5
            color = cmap(shade)
            density_counters[dens] = idx + 1
        plot_amplification_curve(born, label, color)

    report_extreme_files(files)
    plt.xlabel("Cycles")
    plt.ylabel("Amount")
    plt.title("PCR vs Polonies Amplification Curves - All")
    ax = plt.gca()
    ax.grid(True)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.set_xlim(left=0)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    plt.savefig("amplification_curve_mut.svg", format='svg')
    plt.show()
    filepaths_to_fit = glob.glob("results_amplified/polonies*AOE*.csv") + glob.glob("results_amplified/pcr*.csv")
    option = 2
    if option == 1:
        fit_sigmoids_and_save(filepaths_to_fit)
    elif option == 2:
        fit_sigmoids_with_comparisons(filepaths_to_fit)
    compare_sigmoid_rows("sigmoid_fits.csv")


if __name__ == "__main__":
    main()