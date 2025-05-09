#!/usr/bin/env python3
import os, glob, argparse
import pandas as pd
import matplotlib.pyplot as plt

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
    'precision': 's',
    'recall': 'x',
    'specificity': '^'
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

def plot_over_files(df, ax):
    df_pcr = df[df['amplification_type']=='pcr']
    files = df_pcr['file'].tolist()
    idx = range(len(files))
    for m in METRICS:
        ax.scatter(idx, df_pcr[m], color=COLOR_MAP[m], label=m)
    rot = 45 if len(files)>5 else 0
    ax.set_xticks(idx)
    ax.set_xticklabels(files, rotation=rot, ha='right' if rot else 'center')
    ax.set_xlabel('Filename')
    ax.set_ylabel('Metric')
    ax.set_title('All PCR files')
    ax.legend()

def plot_vs_mutation_by_cycle(df, ax, cycle):
    df_pcr = df[df['amplification_type']=='pcr']
    grp = df_pcr[df_pcr['cycles']==cycle]
    vals = grp['mutation_rate'].astype(str).tolist()
    idx = range(len(vals))
    for m in METRICS:
        ax.scatter(idx, grp[m], color=COLOR_MAP[m], label=m)
    rot = 45 if len(vals)>5 else 0
    ax.set_xticks(idx)
    ax.set_xticklabels(vals, rotation=rot, ha='right' if rot else 'center')
    ax.set_xlabel('Mutation rate')
    ax.set_ylabel('Metric')
    ax.set_title(f'Metrics vs Mutation (cycle {cycle})')
    ax.legend()

def plot_vs_cycle_by_mutation(df, ax, mutation_rate):
    df_pcr = df[df['amplification_type']=='pcr']
    grp = df_pcr[df_pcr['mutation_rate']==mutation_rate]
    vals = grp['cycles'].astype(str).tolist()
    idx = range(len(vals))
    for m in METRICS:
        ax.scatter(idx, grp[m], color=COLOR_MAP[m], label=m)
    rot = 45 if len(vals)>5 else 0
    ax.set_xticks(idx)
    ax.set_xticklabels(vals, rotation=rot, ha='right' if rot else 'center')
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Metric')
    ax.set_title(f'Metrics vs Cycle (mutation {mutation_rate})')
    ax.legend()

def plot_metric_vs_mutation_by_cycle(df, ax):
    """
    Combined plot: metrics vs. mutation rate across all cycles.
    """
    df_pcr = df[df['amplification_type'] == 'pcr']
    # Unique, sorted mutation rates for x-axis ticks
    unique_rates = sorted(df_pcr['mutation_rate'].unique())
    rate_strs = [str(r) for r in unique_rates]
    indices = list(range(len(unique_rates)))

    # Scatter each metric at positions of its mutation rates
    for m in METRICS:
        xs, ys = [], []
        for _, row in df_pcr.iterrows():
            xs.append(indices[unique_rates.index(row['mutation_rate'])])
            ys.append(row[m])
        ax.scatter(xs, ys, marker=MARKER_MAP[m], color=COLOR_MAP[m], label=m)

    # X-axis formatting
    rotation = 45 if len(indices) > 5 else 0
    ax.set_xticks(indices)
    ax.set_xticklabels(rate_strs, rotation=rotation, ha='right' if rotation else 'center')
    ax.set_xlabel('Mutation rate')
    ax.set_ylabel('Metric value')
    ax.set_title('Metrics by Mutation rate and Cycle')
    ax.legend(title='Metric')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dir', default='results_metrics')
    args = parser.parse_args()

    df = load_all_metrics(args.dir)

    # First window: All PCR files
    fig1, ax1 = plt.subplots(figsize=(8,6))
    plot_over_files(df, ax1)
    fig1.tight_layout()

    # Second window: Metrics vs Mutation for each cycle
    cycles = sorted(df[df['amplification_type']=='pcr']['cycles'].unique())
    fig2, axes2 = plt.subplots(1, len(cycles), figsize=(14,5))
    for ax, cycle in zip(axes2, cycles):
        plot_vs_mutation_by_cycle(df, ax, cycle)
    fig2.suptitle('Metrics vs Mutation by Cycle')
    fig2.tight_layout()

    # Third window: Metrics vs Cycle for each mutation rate
    mutation_rates = sorted(df[df['amplification_type']=='pcr']['mutation_rate'].unique())
    fig3, axes3 = plt.subplots(1, len(mutation_rates), figsize=(14,5))
    for ax, mut in zip(axes3, mutation_rates):
        plot_vs_cycle_by_mutation(df, ax, mut)
    fig3.suptitle('Metrics vs Cycle by Mutation Rate')
    fig3.tight_layout()

    # Fourth window: Combined metrics by Mutation rate and Cycle
    fig4, ax4 = plt.subplots(figsize=(8,6))
    plot_metric_vs_mutation_by_cycle(df, ax4)
    fig4.tight_layout()

    plt.show()

if __name__=='__main__':
    main()