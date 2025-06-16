#!/usr/bin/env python3

import sys
import os
import csv
import matplotlib.pyplot as plt
import test3

csv.field_size_limit(30 * 1024 * 1024)

def main():
    if len(sys.argv) < 3:
        print("Usage: python test5.py <true_barcodes.csv> <input_files...>")
        sys.exit(1)
    true_csv = sys.argv[1]
    input_files = sys.argv[2:]

    for input_file in input_files:
        groups = test3.group_by_true_barcodes(true_csv, input_file)
        # prepare data
        all_borns = sorted({b for group in groups for b in group['borns']})
        width = 0.8

        # create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
        fig.suptitle(os.path.basename(input_file))

        # --- Plot count distribution with dynamic 5% threshold ---
        seen_labels = set()
        for j, b in enumerate(all_borns):
            group_counts = [(i, grp['borns'].count(b)) for i, grp in enumerate(groups)]
            total_cnt = sum(cnt for _, cnt in group_counts)
            threshold = total_cnt * 0
            above = [(i, cnt) for i, cnt in group_counts if cnt >= threshold]
            below_sum = sum(cnt for _, cnt in group_counts if cnt < threshold)

            bottom = 0
            for idx, cnt in above:
                label = f"Group {idx+1}"
                if label in seen_labels:
                    label = '_nolegend_'
                else:
                    seen_labels.add(label)
                ax1.bar(j, cnt, width, bottom=bottom, label=label)
                bottom += cnt

            if below_sum > 0:
                label = 'below 5% dump group'
                if label in seen_labels:
                    label = '_nolegend_'
                else:
                    seen_labels.add(label)
                ax1.bar(j, below_sum, width, bottom=bottom, label=label)

        ax1.set_xticks(range(len(all_borns)))
        ax1.set_xticklabels(all_borns)
        ax1.set_xlabel('Cycle')
        ax1.set_ylabel('Count')
        ax1.set_title('Cycle-wise ancestry distribution (count)')
        ax1.grid(True)

        # --- Plot percentage distribution per basket ---
        seen_labels = set()
        for j, b in enumerate(all_borns):
            group_counts = [(i, grp['borns'].count(b)) for i, grp in enumerate(groups)]
            total_cnt = sum(cnt for _, cnt in group_counts)
            bottom = 0
            for idx, cnt in group_counts:
                pct = (cnt / total_cnt * 100) if total_cnt > 0 else 0
                label = f"Group {idx+1}"
                if label in seen_labels:
                    label = '_nolegend_'
                else:
                    seen_labels.add(label)
                ax2.bar(j, pct, width, bottom=bottom, label=label)
                bottom += pct

        ax2.set_xticks(range(len(all_borns)))
        ax2.set_xticklabels(all_borns)
        ax2.set_xlabel('Cycle')
        ax2.set_ylabel('Percentage')
        ax2.set_title('Cycle-wise ancestry distribution (percentage)')
        ax2.grid(True)

        # save the page as SVG
        output_svg = os.path.splitext(os.path.basename(input_file))[0] + ".svg"
        fig.savefig(output_svg, format='svg')
        plt.close(fig)

    print("Saved all distribution plots to SVG files")

if __name__ == "__main__":
    main()