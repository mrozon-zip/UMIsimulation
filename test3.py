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
    grp = grp.sort_values(by='mutation_rate')
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
    grp = grp.sort_values(by='cycles')
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

def plot_polonies_over_files(df, ax):
    df_pol = df[df['amplification_type']=='polonies']
    files = df_pol['file'].tolist()
    idx = range(len(files))
    for m in METRICS:
        ax.scatter(idx, df_pol[m], color=COLOR_MAP[m], label=m)
    rot = 45 if len(files)>5 else 0
    ax.set_xticks(idx)
    ax.set_xticklabels(files, rotation=rot, ha='right' if rot else 'center')
    ax.set_xlabel('Filename')
    ax.set_ylabel('Metric')
    ax.set_title('All Polonies files')
    ax.legend()

def plot_metrics_vs_sr(df, ax):
    df_pol = df[df['amplification_type']=='polonies']
    df_pol = df_pol.sort_values(by='Sr')
    vals = df_pol['Sr'].tolist()
    idx = range(len(vals))
    for m in METRICS:
        ax.scatter(idx, df_pol[m], color=COLOR_MAP[m], label=m)
    rot = 45 if len(vals)>5 else 0
    ax.set_xticks(idx)
    ax.set_xticklabels([str(v) for v in vals], rotation=rot, ha='right' if rot else 'center')
    ax.set_xlabel('Sr')
    ax.set_ylabel('Metric')
    ax.set_title('Metrics vs Sr')
    ax.legend()

def plot_metrics_vs_dens(df, ax):
    df_pol = df[df['amplification_type']=='polonies']
    df_pol = df_pol.sort_values(by='dens')
    vals = df_pol['dens'].tolist()
    idx = range(len(vals))
    for m in METRICS:
        ax.scatter(idx, df_pol[m], color=COLOR_MAP[m], label=m)
    rot = 45 if len(vals)>5 else 0
    ax.set_xticks(idx)
    ax.set_xticklabels([str(v) for v in vals], rotation=rot, ha='right' if rot else 'center')
    ax.set_xlabel('Density')
    ax.set_ylabel('Metric')
    ax.set_title('Metrics vs Density')
    ax.legend()

def plot_metrics_vs_mut(df, ax):
    df_pol = df[df['amplification_type']=='polonies']
    df_pol = df_pol.sort_values(by='mut')
    vals = df_pol['mut'].tolist()
    idx = range(len(vals))
    for m in METRICS:
        ax.scatter(idx, df_pol[m], color=COLOR_MAP[m], label=m)
    rot = 45 if len(vals)>5 else 0
    ax.set_xticks(idx)
    ax.set_xticklabels([str(v) for v in vals], rotation=rot, ha='right' if rot else 'center')
    ax.set_xlabel('Mutation rate')
    ax.set_ylabel('Metric')
    ax.set_title('Metrics vs Mutation rate')
    ax.legend()

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

    # Polonies windows
    # Window 1: All Polonies files
    fig5, ax5 = plt.subplots(figsize=(8,6))
    plot_polonies_over_files(df, ax5)
    fig5.tight_layout()

    # Prepare polonies subset and values
    df_pol = df[df['amplification_type']=='polonies']
    sr_vals   = sorted(df_pol['Sr'].unique())
    mut_vals  = sorted(df_pol['mut'].unique())
    dens_vals = sorted(df_pol['dens'].unique())
    print("mut_vals:", mut_vals)

    # Default constants
    sr_const  = sr_vals[0]
    mut_const = mut_vals[0]
    dens_const= dens_vals[0]

    # Window 2: Metrics vs Sr (row1: dens_const & mut_vals; row2: mut_const & dens_vals)
    ncols = max(len(mut_vals), len(dens_vals))
    fig6, axes6 = plt.subplots(2, ncols, figsize=(4*ncols, 8), squeeze=False)
    # Row 1: vary mutation_rate at fixed density
    for ax, mut in zip(axes6[0], mut_vals):
        grp = df_pol[(df_pol['dens']==dens_const) & (df_pol['mut']==mut)]
        grp = grp.sort_values(by='Sr')
        x = list(range(len(grp)))
        vals = grp['Sr'].astype(str).tolist()
        for m in METRICS:
            ax.scatter(x, grp[m], marker=MARKER_MAP[m], color=COLOR_MAP[m], label=m)
        ax.set_xticks(x)
        ax.set_xticklabels(vals, rotation=45, ha='right')
        ax.set_xlabel('Sr')
        ax.set_ylabel('Metric')
        ax.set_title(f'dens={dens_const}, mut={mut}')
        ax.legend()
    for ax in axes6[0][len(mut_vals):]:
        ax.axis('off')
    # Row 2: vary density at fixed mutation_rate
    for ax, dens in zip(axes6[1], dens_vals):
        grp = df_pol[(df_pol['mut']==mut_const) & (df_pol['dens']==dens)]
        grp = grp.sort_values(by='Sr')
        x = list(range(len(grp)))
        vals = grp['Sr'].astype(str).tolist()
        for m in METRICS:
            ax.scatter(x, grp[m], marker=MARKER_MAP[m], color=COLOR_MAP[m], label=m)
        ax.set_xticks(x)
        ax.set_xticklabels(vals, rotation=45, ha='right')
        ax.set_xlabel('Sr')
        ax.set_ylabel('Metric')
        ax.set_title(f'mut={mut_const}, dens={dens}')
        ax.legend()
    for ax in axes6[1][len(dens_vals):]:
        ax.axis('off')
    fig6.suptitle('Metrics vs Sr')
    fig6.tight_layout()

    # Window 3: Metrics vs Density (row1: sr_const & mut_vals; row2: mut_const & sr_vals)
    ncols = max(len(mut_vals), len(sr_vals))
    fig7, axes7 = plt.subplots(2, ncols, figsize=(4*ncols, 8), squeeze=False)
    # Row 1: vary mutation_rate at fixed Sr
    for ax, mut in zip(axes7[0], mut_vals):
        grp = df_pol[(df_pol['Sr']==sr_const) & (df_pol['mut']==mut)]
        grp = grp.sort_values(by='dens')
        x = list(range(len(grp)))
        vals = grp['dens'].astype(str).tolist()
        for m in METRICS:
            ax.scatter(x, grp[m], marker=MARKER_MAP[m], color=COLOR_MAP[m], label=m)
        ax.set_xticks(x)
        ax.set_xticklabels(vals, rotation=45, ha='right')
        ax.set_xlabel('Density')
        ax.set_ylabel('Metric')
        ax.set_title(f'Sr={sr_const}, mut={mut}')
        ax.legend()
    for ax in axes7[0][len(mut_vals):]:
        ax.axis('off')
    # Row 2: vary Sr at fixed mutation_rate
    for ax, sr in zip(axes7[1], sr_vals):
        grp = df_pol[(df_pol['mut']==mut_const) & (df_pol['Sr']==sr)]
        grp = grp.sort_values(by='dens')
        x = list(range(len(grp)))
        vals = grp['dens'].astype(str).tolist()
        for m in METRICS:
            ax.scatter(x, grp[m], marker=MARKER_MAP[m], color=COLOR_MAP[m], label=m)
        ax.set_xticks(x)
        ax.set_xticklabels(vals, rotation=45, ha='right')
        ax.set_xlabel('Density')
        ax.set_ylabel('Metric')
        ax.set_title(f'mut={mut_const}, Sr={sr}')
        ax.legend()
    for ax in axes7[1][len(sr_vals):]:
        ax.axis('off')
    fig7.suptitle('Metrics vs Density')
    fig7.tight_layout()

    # Window 4: Metrics vs Mutation rate (row1: sr_const & dens_vals; row2: dens_const & sr_vals)
    ncols = max(len(dens_vals), len(sr_vals))
    fig8, axes8 = plt.subplots(2, ncols, figsize=(4*ncols, 8), squeeze=False)
    # Row 1: vary density at fixed Sr
    for ax, dens in zip(axes8[0], dens_vals):
        grp = df_pol[(df_pol['Sr']==sr_const) & (df_pol['dens']==dens)]
        grp = grp.sort_values(by='mut')
        x = list(range(len(grp)))
        vals = grp['mut'].astype(str).tolist()
        for m in METRICS:
            ax.scatter(x, grp[m], marker=MARKER_MAP[m], color=COLOR_MAP[m], label=m)
        ax.set_xticks(x)
        ax.set_xticklabels(vals, rotation=45, ha='right')
        ax.set_xlabel('Mutation rate')
        ax.set_ylabel('Metric')
        ax.set_title(f'Sr={sr_const}, dens={dens}')
        ax.legend()
    for ax in axes8[0][len(dens_vals):]:
        ax.axis('off')
    # Row 2: vary Sr at fixed density
    for ax, sr in zip(axes8[1], sr_vals):
        grp = df_pol[(df_pol['dens']==dens_const) & (df_pol['Sr']==sr)]
        grp = grp.sort_values(by='mut')
        x = list(range(len(grp)))
        vals = grp['mut'].astype(str).tolist()
        for m in METRICS:
            ax.scatter(x, grp[m], marker=MARKER_MAP[m], color=COLOR_MAP[m], label=m)
        ax.set_xticks(x)
        ax.set_xticklabels(vals, rotation=45, ha='right')
        ax.set_xlabel('Mutation rate')
        ax.set_ylabel('Metric')
        ax.set_title(f'dens={dens_const}, Sr={sr}')
        ax.legend()
    for ax in axes8[1][len(sr_vals):]:
        ax.axis('off')
    fig8.suptitle('Metrics vs Mutation rate')
    fig8.tight_layout()

    # New window: Boxplots for PCR vs Polonies metrics
    fig9, ax9 = plt.subplots(figsize=(10,6))
    # Prepare data for each amplification type and metric
    df_pcr = df[(df['amplification_type']=='pcr') & (df['mutation_rate'] != 0)]
    df_pol = df[(df['amplification_type']=='polonies') & (df['mut'] != 0)]
    data = []
    labels = []
    colors = []
    for amp, subset in [('pcr', df_pcr), ('polonies', df_pol)]:
        for m in METRICS:
            data.append(subset[m].dropna().tolist())
            labels.append(f'{amp} {m}')
            colors.append(COLOR_MAP[m])
    # Create colored boxplots
    bp = ax9.boxplot(data, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    # Format axes
    ax9.set_xticklabels(labels, rotation=45, ha='right')
    ax9.set_xlabel('Amplification type and metric')
    ax9.set_ylabel('Metric value')
    ax9.set_title('Distribution of metrics for PCR and Polonies')
    # Legend for metric colors
    handles = [Patch(facecolor=COLOR_MAP[m], label=m) for m in METRICS]
    ax9.legend(handles=handles, title='Metric')
    fig9.tight_layout()

    plt.show()

if __name__=='__main__':
    main()