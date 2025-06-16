import csv
import re
import matplotlib.pyplot as plt

# --- New helper and grouping functions ---
def _parse_int_list(s) -> list[int]:
    """
    Parse a value that may be a string like "[1, 0, 2]" or an actual list/tuple of ints.
    """
    if isinstance(s, (list, tuple)):
        return [int(x) for x in s]
    s_str = str(s)
    return list(map(int, re.findall(r'\d+', s_str)))

def group_by_true_barcodes(true_csv: str, input_csv: str):
    """
    For each row in input_csv whose 'sequence' is in true_csv,
    build a separate group seeded by that row. Then iteratively
    add any other rows whose parent_id list intersects the group's
    current id set. Each group stores only its accumulated born list.
    Finally, print:
      - X out of Y found
      - Number of groups
      - Length of each group's born list.
    """
    # Load true barcode sequences
    true_seqs = set()
    with open(true_csv, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            true_seqs.add(row['sequence'])
    Y = len(true_seqs)

    # Load all rows from input file
    input_rows = []
    with open(input_csv, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            input_rows.append({
                'sequence': row['sequence'],
                'ids': _parse_int_list(row['id']),
                'parent_ids': _parse_int_list(row['parent_id']),
                'borns': _parse_int_list(row['born'])
            })
    # compute total born elements across all input rows
    check = sum(len(row['borns']) for row in input_rows)
    print(f"Total born elements in input file: {check}")

    # Sanity checks
    print(f"Rows in input: {len(input_rows)}")
    seed_count = sum(1 for row in input_rows if row['sequence'] in true_seqs)
    print(f"Seeds matched: {seed_count} out of {Y}")


    # Seed groups and remember exactly which row-indices we used
    seeded_indices = []
    for idx, row in enumerate(input_rows):
        if row['sequence'] in true_seqs:
            seeded_indices.append(idx)
    # X is number of unique true barcodes found in the input
    X = len(seeded_indices)

    # Build a single group per seed up-front
    unique_groups = {}
    for idx in seeded_indices:
        seed = input_rows[idx]['sequence']
        if seed not in unique_groups:
            unique_groups[seed] = {
                'ids': set(input_rows[idx]['ids']),
                'borns': list(input_rows[idx]['borns']),
                'seed_sequence': seed
            }
    groups = list(unique_groups.values())

    # Initialize “assigned” to exactly those rows we seeded
    assigned = set(seeded_indices)
    changed = True
    while changed:
        changed = False
        for idx, row in enumerate(input_rows):
            if idx in assigned:
                continue
            for group in groups:
                if any(pid in group['ids'] for pid in row['parent_ids']):
                    group['ids'].update(row['ids'])
                    group['borns'].extend(row['borns'])
                    assigned.add(idx)
                    changed = True
                    break

    # Print summary
    print(f"Found {X} out of {Y} true barcodes in input file")
    print(f"Number of groups: {len(groups)}")


    # Merge groups with duplicate seed sequences by rebuilding list
    merged_groups = []
    seen_seeds = set()
    for group in groups:
        seed = group['seed_sequence']
        if seed not in seen_seeds:
            # collect all groups with this seed
            duplicates = [g for g in groups if g['seed_sequence'] == seed]
            # merge details
            merged_group = {
                'ids': set(),
                'borns': [],
                'seed_sequence': seed
            }
            for g in duplicates:
                merged_group['ids'].update(g['ids'])
                merged_group['borns'].extend(g['borns'])
            merged_groups.append(merged_group)
            seen_seeds.add(seed)
    groups = merged_groups

    for i, group in enumerate(groups, start=1):
        print(f"Group {i} (seed: {group['seed_sequence']}): born list length {len(group['borns'])}")

    # Total born elements across all groups
    total_born_elements = sum(len(group['borns']) for group in groups)
    print(f"Total born elements across all groups: {total_born_elements}")

    # Check for duplicate sequences across groups
    return groups

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python test3.py <true_barcodes.csv> <input_file.csv>")
        sys.exit(1)

    true_csv = sys.argv[1]
    input_csv = sys.argv[2]
    groups = group_by_true_barcodes(true_csv, input_csv)
    # collect unique born values
    all_borns = sorted({b for group in groups for b in group['borns']})
    width = 0.8

    # plot stacked bars per basket: groups with ≥100 counts separately, others aggregated
    seen_labels = set()
    for j, b in enumerate(all_borns):
        # compute each group's count at this born value
        group_counts = [(i, group['borns'].count(b)) for i, group in enumerate(groups)]
        # compute dynamic 5% threshold based on this basket's total
        total_cnt = sum(cnt for _, cnt in group_counts)
        print(total_cnt)
        threshold = total_cnt * 0.03
        above = [(i, cnt) for i, cnt in group_counts if cnt >= threshold]
        below_sum = sum(cnt for _, cnt in group_counts if cnt < threshold)

        bottom = 0
        # plot heavy groups
        for idx, cnt in above:
            label = f"Group {idx+1}"
            if label in seen_labels:
                label = '_nolegend_'
            else:
                seen_labels.add(label)
            plt.bar(j, cnt, width, bottom=bottom, label=label)
            bottom += cnt

        # plot below-threshold dump group
        if below_sum > 0:
            label = 'below 5% dump group'
            if label in seen_labels:
                label = '_nolegend_'
            else:
                seen_labels.add(label)
            plt.bar(j, below_sum, width, bottom=bottom, label=label)

    plt.xticks(range(len(all_borns)), all_borns)
    plt.xlabel('born value')
    plt.ylabel('count')
    plt.title('born-wise ancestry distribution')
    #plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot percentage distribution of born values per basket by group
    plt.figure()
    width = 0.8
    seen_labels = set()
    for j, b in enumerate(all_borns):
        # count per group
        group_counts = [(i, group['borns'].count(b)) for i, group in enumerate(groups)]
        total_cnt = sum(cnt for _, cnt in group_counts)
        bottom = 0
        for i, cnt in group_counts:
            pct = (cnt / total_cnt * 100) if total_cnt > 0 else 0
            label = f"Group {i+1}"
            if label in seen_labels:
                label = '_nolegend_'
            else:
                seen_labels.add(label)
            plt.bar(j, pct, width, bottom=bottom, label=label)
            bottom += pct
    plt.xticks(range(len(all_borns)), all_borns)
    plt.xlabel('born value')
    plt.ylabel('percentage')
    plt.title('born-wise ancestry percentage distribution')
    #plt.legend()
    plt.tight_layout()
    plt.show()