#!/usr/bin/env python3
import os, glob, argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

METRICS = ['accuracy','precision','recall','specificity']

# Define colors and markers for metrics (replace default orange with purple)
COLOR_MAP = {
    'accuracy': 'blue',
    'precision': 'purple',
    'recall': 'green',
    'specificity': 'red'
}
MARKER_MAP = {
    'accuracy': 'o',
    'precision': 'o',
    'recall': 'o',
    'specificity': 'o'
}

def load_all_metrics(metrics_dir):
    pattern = os.path.join(metrics_dir, '*.csv')
    files = glob.glob(pattern)
    dfs = []
    for p in files:
        df = pd.read_csv(p)
        df['file'] = os.path.basename(p)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dir', default='results_metrics')
    args = parser.parse_args()

    df = load_all_metrics(args.dir)

    # Generate tables for files by amplification type
    df_pcr_files = df[df['file'].str.contains('pcr')]
    # For PCR metrics table: drop unwanted columns
    df_pcr_files = df_pcr_files.drop(columns=['mut', 'Sr', 'dens', 'AOE'])
    df_pol_files = df[df['file'].str.contains('polonies')]
    # For polonies metrics table: drop unwanted columns and rename 'mut'
    df_pol_files = df_pol_files.drop(columns=['mutation_rate', 'cycles'])
    df_pol_files = df_pol_files.rename(columns={'mut': 'mutation rate'})

    print("\n=== PCR Files Metrics Table ===")
    print(df_pcr_files.to_string(index=False))

    print("\n=== Polonies Files Metrics Table ===")
    print(df_pol_files.to_string(index=False))

    # Optionally save the tables to CSV files
    df_pcr_files.to_csv('pcr_metrics_table3.csv', index=False)
    df_pol_files.to_csv('polonies_metrics_table3.csv', index=False)

    # Compute extremes per amplification type
    for amp in ['pcr', 'polonies']:
        df_amp = df[df['amplification_type'] == amp]
        if df_amp.empty:
            continue
        print(f"\n=== {amp.upper()} ===")
        # Precision
        max_prec_idx = df_amp['precision'].idxmax()
        file_max_prec = df_amp.loc[max_prec_idx, 'file']
        print(f"Highest precision for {amp}: {file_max_prec} ({df_amp.loc[max_prec_idx, 'precision']})")
        min_prec_idx = df_amp['precision'].idxmin()
        file_min_prec = df_amp.loc[min_prec_idx, 'file']
        print(f"Lowest precision for {amp}: {file_min_prec} ({df_amp.loc[min_prec_idx, 'precision']})")
        # Accuracy
        max_acc_idx = df_amp['accuracy'].idxmax()
        file_max_acc = df_amp.loc[max_acc_idx, 'file']
        print(f"Highest accuracy for {amp}: {file_max_acc} ({df_amp.loc[max_acc_idx, 'accuracy']})")
        min_acc_idx = df_amp['accuracy'].idxmin()
        file_min_acc = df_amp.loc[min_acc_idx, 'file']
        print(f"Lowest accuracy for {amp}: {file_min_acc} ({df_amp.loc[min_acc_idx, 'accuracy']})")
        # Combined precision and accuracy
        df_amp = df_amp.copy()
        df_amp['prec_acc_sum'] = df_amp['precision'] + df_amp['accuracy']
        max_sum_idx = df_amp['prec_acc_sum'].idxmax()
        file_max_sum = df_amp.loc[max_sum_idx, 'file']
        print(f"Highest combined precision and accuracy for {amp}: {file_max_sum} prec: ({df_amp.loc[max_sum_idx, 'precision']}) acc: ({df_amp.loc[max_sum_idx, 'accuracy']})")
        min_sum_idx = df_amp['prec_acc_sum'].idxmin()
        file_min_sum = df_amp.loc[min_sum_idx, 'file']
        print(f"Lowest combined precision and accuracy for {amp}: {file_min_sum} prec: ({df_amp.loc[min_sum_idx, 'precision']}) acc: ({df_amp.loc[min_sum_idx, 'accuracy']})")


    # New window: Boxplots for PCR vs Polonies metrics split into two subplots
    fig9, (ax9_pcr, ax9_pol) = plt.subplots(1, 2, figsize=(12,6))
    fig9.suptitle('Metrics Distribution. Threshold = 10')

    # Prepare data for PCR metrics
    df_pcr = df[(df['amplification_type']=='pcr') & (df['mutation_rate'] != 0)]
    pcr_data = [df_pcr[m].dropna().tolist() for m in METRICS]
    bp_pcr = ax9_pcr.boxplot(pcr_data, patch_artist=True)
    for patch, m in zip(bp_pcr['boxes'], METRICS):
        patch.set_facecolor(COLOR_MAP[m])
    ax9_pcr.set_xticklabels(METRICS, rotation=45, ha='right')
    ax9_pcr.set_xlabel('Metric')
    ax9_pcr.set_ylabel('Metric value')
    ax9_pcr.set_title('PCR Metrics')

    # Prepare data for Polonies metrics
    df_pol = df[(df['amplification_type']=='polonies') & (df['mut'] != 0)]
    pol_data = [df_pol[m].dropna().tolist() for m in METRICS]
    bp_pol = ax9_pol.boxplot(pol_data, patch_artist=True)
    for patch, m in zip(bp_pol['boxes'], METRICS):
        patch.set_facecolor(COLOR_MAP[m])
    ax9_pol.set_xticklabels(METRICS, rotation=45, ha='right')
    ax9_pol.set_xlabel('Metric')
    ax9_pol.set_ylabel('Metric value')
    ax9_pol.set_title('Polonies Metrics')

    fig9.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig('metrics3.svg', format='svg')
    plt.show()

if __name__=='__main__':
    main()