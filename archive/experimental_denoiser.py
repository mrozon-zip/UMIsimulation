import sys
import csv
import ast
import pandas as pd
from pathlib import Path
from collections import defaultdict, deque

csv.field_size_limit(30 * 1024 * 1024)

# Default maximum edit distance for connections
DEFAULT_MAX_ED = 10

# Try to use C-based Levenshtein if available
try:
    from Levenshtein import distance as _ldistance
    def edit_distance(s: str, t: str, max_dist: int) -> int:
        """C-based edit distance; full distance (no threshold early exit)"""
        return _ldistance(s, t)
    FAST_ED = True
except ImportError:
    FAST_ED = False
    def edit_distance(s: str, t: str, max_dist: int) -> int:
        """Thresholded Python edit distance with early exit beyond max_dist."""
        m, n = len(s), len(t)
        if abs(m - n) > max_dist:
            return max_dist + 1
        if m > n:
            return edit_distance(t, s, max_dist)
        prev = list(range(n + 1))
        for i in range(1, m + 1):
            cur = [i] + [0] * n
            row_min = cur[0]
            for j in range(1, n + 1):
                ins = cur[j - 1] + 1
                rem = prev[j] + 1
                sub = prev[j - 1] + (s[i - 1] != t[j - 1])
                cur[j] = min(ins, rem, sub)
                row_min = min(row_min, cur[j])
            if row_min > max_dist:
                return max_dist + 1
            prev = cur
        return prev[n]

class BKNode:
    """Node in a BK-tree storing an index into the sequences list."""
    __slots__ = ('idx', 'children')

    def __init__(self, idx: int):
        self.idx = idx
        self.children = {}  # distance -> BKNode

    def add(self, idx: int, sequences: list, max_ed: int):
        """Add a new index to the BK-tree using threshold distance."""
        dist = edit_distance(sequences[self.idx], sequences[idx], max_ed)
        # only add if distance <= max_ed * 2 (to limit branching)
        if dist > max_ed * 2:
            return
        child = self.children.get(dist)
        if child is None:
            self.children[dist] = BKNode(idx)
        else:
            child.add(idx, sequences, max_ed)

    def search(self, idx: int, sequences: list, max_ed: int, results: list):
        """Find all indices whose sequence is within max_ed of sequences[idx]."""
        node_seq = sequences[self.idx]
        query_seq = sequences[idx]
        # length pre-filter
        if abs(len(node_seq) - len(query_seq)) > max_ed:
            # possible distances in subtree range from |D - child_dist| to D + child_dist,
            # skip only this node but still explore children if child_dist could bring it in range
            pass
        dist = edit_distance(node_seq, query_seq, max_ed)
        if dist <= max_ed:
            results.append(self.idx)
        # explore children within metric bounds
        lower, upper = dist - max_ed, dist + max_ed
        for d, child in self.children.items():
            if lower <= d <= upper:
                child.search(idx, sequences, max_ed, results)


def denoise_csv(input_csv: str, max_edit_distance: int = DEFAULT_MAX_ED) -> str:
    """
    Read `input_csv`, build directed graph where A->B if:
      1) N0[A] >= 2 * N0[B]
      2) edit_distance(seq(A), seq(B)) <= max_edit_distance
    Uses a BK-tree for neighbor queries, optimized with an optional C-based Levenshtein
    and early filtering on sequence length.

    Then for each weakly-connected component, identify the central node
    (node with zero incoming edges; if ties, pick highest N0), sum all N0
    in the component, and write out only those central sequences with
    cumulated N0 to `folderB/<basename>_denoised.csv`.

    Returns the path to the written output CSV.
    """
    infile = Path(input_csv)
    if not infile.exists():
        raise FileNotFoundError(f"Input file not found: {infile}")

    # Derive output path
    stem = infile.stem
    out_name = f"{stem}_denoised.csv"
    parent = infile.parent
    if parent.name == "folderA":
        out_parent = parent.parent / "folderB"
    else:
        out_parent = parent / "folderB"
    out_parent.mkdir(parents=True, exist_ok=True)
    outfile = out_parent / out_name

    # Read data
    df = pd.read_csv(infile, dtype={"sequence": str})
    if "sequence" not in df.columns or "N0" not in df.columns:
        raise ValueError("CSV must contain 'sequence' and 'N0' columns")

    sequences = df["sequence"].tolist()
    counts = df["N0"].astype(int).tolist()
    n = len(sequences)
    if n == 0:
        pd.DataFrame(columns=["sequence", "N0"]).to_csv(outfile, index=False)
        return str(outfile)

    # Build BK-tree
    root = BKNode(0)
    for i in range(1, n):
        root.add(i, sequences, max_edit_distance)

    # Build directed & undirected graphs
    graph_dir = defaultdict(list)
    graph_undir = defaultdict(set)

    for i in range(n):
        neighbors = []
        root.search(i, sequences, max_edit_distance, neighbors)
        for j in set(neighbors):
            if j == i or j < i:
                continue
            cnt_i, cnt_j = counts[i], counts[j]
            if cnt_i >= 2 * cnt_j:
                graph_dir[i].append(j)
            if cnt_j >= 2 * cnt_i:
                graph_dir[j].append(i)
            graph_undir[i].add(j)
            graph_undir[j].add(i)

    # Find weakly-connected components and central nodes
    visited = set()
    output = []  # (sequence, total_count)

    for node in range(n):
        if node in visited:
            continue
        comp = []
        queue = deque([node])
        visited.add(node)
        while queue:
            u = queue.popleft()
            comp.append(u)
            for v in graph_undir[u]:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)

        indegree = {u: 0 for u in comp}
        for u in comp:
            for v in graph_dir.get(u, []):
                if v in indegree:
                    indegree[v] += 1

        centrals = [u for u, d in indegree.items() if d == 0]
        if not centrals:
            continue
        central = max(centrals, key=lambda u: counts[u])
        total_count = sum(counts[u] for u in comp)
        output.append((sequences[central], total_count))

    # Write output
    df_out = pd.DataFrame(output, columns=["sequence", "N0"])
    df_out.to_csv(outfile, index=False)
    return str(outfile)

def build_sequence_network(csv_file_path):
    """
    Reads a CSV where each row has:
        sequence, id="[…]", parent_id="[…]", …
    and builds:
      1. network: dict mapping parent_sequence -> [child_sequence, …]
      2. edit_distance_hist: {distance: count, …}
      3. edit_distances: list of all distances

    This version skips any link where the parent_id appears in the same row
    as the child_id (i.e. ignores “within-the-same-row” connections).
    """
    # --- Levenshtein distance (Wagner–Fischer) ---
    def levenshtein_distance(s1, s2):
        m, n = len(s1), len(s2)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m+1): dp[i][0] = i
        for j in range(n+1): dp[0][j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                cost = 0 if s1[i-1]==s2[j-1] else 1
                dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
        return dp[m][n]

    # --- Read CSV and map each ID to (sequence, row_index) ---
    id_to_seq = {}
    id_to_row = {}
    rows = []  # will store (row_index, child_ids, parent_ids, child_seq)

    with open(csv_file_path, newline='') as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            seq = row['sequence']
            child_ids  = ast.literal_eval(row['id'])
            parent_ids = ast.literal_eval(row['parent_id'])
            if len(child_ids) != len(parent_ids):
                raise ValueError(f"Row {row_idx}: id/parent_id length mismatch")

            # Record for each child_id which row it came from
            for cid in child_ids:
                if cid in id_to_seq and id_to_seq[cid] != seq:
                    raise ValueError(f"ID {cid} redefined on row {row_idx}")
                id_to_seq[cid] = seq
                id_to_row[cid] = row_idx

            rows.append((row_idx, child_ids, parent_ids, seq))

    # --- Build network and gather distances ---
    network = {}
    edit_distances = []

    for row_idx, child_ids, parent_ids, child_seq in rows:
        for cid, pid in zip(child_ids, parent_ids):
            # 1) skip self-links
            if cid == pid:
                continue
            # 2) skip any link where parent_id was defined on the same row
            if id_to_row.get(pid) == row_idx:
                continue
            # 3) skip if parent_id not in our map
            parent_seq = id_to_seq.get(pid)
            if parent_seq is None:
                continue

            # record the edge
            network.setdefault(parent_seq, []).append(child_seq)

            # compute & record distance
            d = levenshtein_distance(parent_seq, child_seq)
            edit_distances.append(d)

    # --- Histogram of distances ---
    edit_distance_hist = {}
    for d in edit_distances:
        edit_distance_hist[d] = edit_distance_hist.get(d, 0) + 1

    return network, edit_distance_hist, edit_distances

if __name__ == "__main__":
    import glob
    from collections import Counter

    hist_total = Counter()
    # Find all files in results_amplidied with "pcr" in filename
    files = glob.glob("../results_amplified/polonies_mut_0.01_Sr_30_dens_100_AOE_10.csv")
    if not files:
        print("No files with 'pcr' in filename found in results_amplidied.")
        sys.exit(1)
    for f in files:
        print(f"Processing {f}...")
        _, hist, _ = build_sequence_network(f)
        hist_total.update(hist)
    # Save the total histogram to a CSV file
    out_csv = "total_edit_distance_hist2.csv"
    with open(out_csv, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["levenshtein_distance", "count"])
        for dist in sorted(hist_total):
            writer.writerow([dist, hist_total[dist]])
    print(f"Saved accumulated edit distance histogram to {out_csv}")