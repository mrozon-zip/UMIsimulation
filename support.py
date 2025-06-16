import math
from typing import Dict, Tuple, List, Union
import os
import csv
import random
import pandas as pd
from pathlib import Path
import sys
from Levenshtein import distance

random.seed(42)

# Global nucleotides list
NUCLEOTIDES = ['A', 'C', 'G', 'T']


def denoise(input_csv_path, treshold):
    """
    Reads a CSV, identifies 'central' nodes, and writes them to a new CSV.

    A node X is considered 'central' if no other node Y exists such that:
    1. count(Y) >= 2 * count(X)
    2. levenshtein_distance(sequence(Y), sequence(X)) == 1

    Args:
        input_csv_path (str): Path to the input CSV file.
        output_dir (str): Path to the directory where the output CSV will be saved.
    """
    try:
        # Read the input CSV
        df = pd.read_csv(input_csv_path)

        # Validate required columns
        if 'sequence' not in df.columns or 'N0' not in df.columns:
            print(f"Error: Input CSV '{input_csv_path}' must contain 'sequence' and 'N0' columns.", file=sys.stderr)
            return

        # Ensure N0 is numeric, handle potential errors
        try:
            df['N0'] = pd.to_numeric(df['N0'])
        except ValueError:
            print(f"Error: Column 'N0' in '{input_csv_path}' contains non-numeric values.", file=sys.stderr)
            return

        # Convert to list of dictionaries for potentially faster access? Or use iloc/loc
        # Using df.itertuples() is generally efficient
        nodes = list(df.itertuples())

        indices_to_keep = []

        print(f"Processing {len(nodes)} nodes from {input_csv_path}...")

        # Iterate through each node (X) to determine if it's 'central'
        for i, node_x in enumerate(nodes):
            is_potentially_erroneous = False
            seq_x = node_x.sequence
            count_x = node_x.N0

            # Compare node X with every other node Y
            for j, node_y in enumerate(nodes):
                if i == j:  # Don't compare a node to itself
                    continue

                seq_y = node_y.sequence
                count_y = node_y.N0

                # Check the conditions for X being potentially erroneous due to Y
                # Condition 1: Node Y count is significantly higher
                count_condition = (count_y >= 2 * count_x)

                # Condition 2: Sequences are 1 edit distance apart
                # Only calculate distance if count condition is met (optimization)
                if count_condition:
                    edit_distance = distance(seq_y, seq_x)
                    if edit_distance <= treshold:
                        is_potentially_erroneous = True
                        # print(f"Node {i} ('{seq_x}', {count_x}) is potentially erroneous due to Node {j} ('{seq_y}', {count_y})")
                        break  # Found a node Y that makes X non-central, no need to check others

            # If after checking all other nodes Y, no condition was met, keep node X
            if not is_potentially_erroneous:
                indices_to_keep.append(node_x.Index)  # Use the original DataFrame index

        print(f"Identified {len(indices_to_keep)} central nodes.")

        # Filter the original DataFrame
        df_denoised = df.loc[indices_to_keep]

        # --- Output File Path Handling ---
        input_path_obj = Path(input_csv_path)
        input_filename_stem = input_path_obj.stem  # Filename without extension
        output_filename = f"{input_filename_stem}_type2_denoised.csv"

        output_path_obj = Path("results_denoised")
        # Create the output directory if it doesn't exist
        output_path_obj.mkdir(parents=True, exist_ok=True)

        output_csv_path = output_path_obj / output_filename
        # --- ---

        # Write the denoised data to the output CSV
        df_denoised.to_csv(output_csv_path, index=False)
        print(f"Denoised data saved to: {output_csv_path}")

    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_csv_path}'", file=sys.stderr)
    except pd.errors.EmptyDataError:
        print(f"Error: Input file '{input_csv_path}' is empty.", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)


def csv_to_list_of_dicts(filename):
    """
    Reads a CSV file and converts its contents into a list of dictionaries.
    Each dictionary corresponds to a row in the CSV, with keys taken from the header.
    """
    data = []
    with open(filename, newline='', encoding='utf-8') as csvfile:
        # DictReader automatically uses the first row as header
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data


def filter_data(data):
    """
    Filters a list of dictionaries to keep only the 'sequence' and 'N0' keys.

    Parameters:
        data (list): A list of dictionaries, each representing a row from the CSV.

    Returns:
        list: A new list of dictionaries containing only the 'sequence' and 'N0' keys.
    """
    filtered = []
    for entry in data:
        filtered.append({
            "sequence": entry.get("sequence"),
            "N0": entry.get("N0")
        })
    return filtered


def levenshtein_distance(s1, s2):
    """
    Computes the Levenshtein distance between two strings s1 and s2.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def assign_connections(data):
    """
    Assigns a new key 'connections' to each dictionary in the list 'data'.
    The value is a list of indices of dictionaries that are connected.

    Connection criteria:
    - Two dictionaries are connected if the Levenshtein distance between their 'sequence' values is 1.
    - Additionally, dictionary A connects to dictionary B if the N0 value of dictionary A is
      higher or equal to 2*(N0 of dictionary B - 1).
    """
    for i, entry_a in enumerate(data):
        connections = []
        try:
            n0_a = int(entry_a.get("N0", 0))
        except ValueError:
            n0_a = 0
        for j, entry_b in enumerate(data):
            if i == j:
                continue
            try:
                n0_b = int(entry_b.get("N0", 0))
            except ValueError:
                n0_b = 0
            # Check N0 condition: dictionary A's N0 is higher or equal to 2*(N0 of dictionary B - 1)
            if n0_a >= 2 * (n0_b - 1):
                seq_a = entry_a.get("sequence", "")
                seq_b = entry_b.get("sequence", "")
                # Check if sequences are one edit distance apart
                if levenshtein_distance(seq_a, seq_b) == 1:
                    connections.append(j)
        entry_a["connections"] = connections


def group_connected_components(data):
    """
    Groups dictionaries into connected components based on their 'connections' field.
    Returns a list of groups, where each group is a list of indices referring to dictionaries in the input list.
    """
    # Build graph as an undirected graph
    graph = {i: set() for i in range(len(data))}
    for i, entry in enumerate(data):
        for j in entry.get("connections", []):
            graph[i].add(j)
            graph[j].add(i)

    visited = set()
    groups = []

    def dfs(node, group):
        visited.add(node)
        group.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor, group)

    for i in range(len(data)):
        if i not in visited:
            group = []
            dfs(i, group)
            groups.append(group)

    return groups


def collapse_groups(data, groups):
    """
    For each group, collapse the dictionaries to the one with the highest N0.
    If there is a tie for N0 value and it equals 2, pick one randomly.
    Returns a list of collapsed dictionaries, each containing only the 'sequence' and 'N0' fields.
    """
    collapsed = []
    for group in groups:
        # Extract group items
        group_items = [data[i] for i in group]
        # Convert N0 values to int (defaulting to 0 on failure)
        group_items_with_n0 = [(item, int(item.get("N0", "0"))) for item in group_items]
        # Sum N0 values for the group
        total_n0 = sum(n0 for _, n0 in group_items_with_n0)
        # Find the maximum N0
        max_n0 = max(n0 for _, n0 in group_items_with_n0)
        candidates = [item for item, n0 in group_items_with_n0 if n0 == max_n0]
        # If tie and value equals 2, choose one randomly
        if len(candidates) > 1 and max_n0 == 2:
            chosen = random.choice(candidates)
        else:
            chosen = candidates[0]
        # Create a new collapsed dictionary with updated N0 (as string) and the same sequence
        chosen_updated = {"sequence": chosen.get("sequence", ""), "N0": str(total_n0)}
        collapsed.append(chosen_updated)
    return collapsed


def write_collapsed_csv(filename, collapsed_data):
    """
    Writes the collapsed dictionaries to a new CSV file.
    The new file is named by appending '_denoised' before the file extension of the input filename.
    Only the 'sequence' and 'N0' columns are included in the output.
    """
    base, ext = os.path.splitext(filename)
    out_filename = base + "_denoised" + ext
    with open(out_filename, mode="w", newline='', encoding="utf-8") as csvfile:
        fieldnames = ["sequence", "N0"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in collapsed_data:
            writer.writerow(row)


def process_file(filename):
    """
    Processes the input CSV file by performing the following steps:
    - Reads the CSV into a list of dictionaries.
    - Filters data to retain only 'sequence' and 'N0'.
    - Assigns connections (using the existing assign_connections function).
    - Groups connected components.
    - Collapses each group to a single dictionary.
    - Writes the collapsed dictionaries to a new CSV file.
    """
    data = csv_to_list_of_dicts(filename)
    filtered_data = filter_data(data)
    assign_connections(filtered_data)
    groups = group_connected_components(filtered_data)
    collapsed = collapse_groups(filtered_data, groups)
    write_collapsed_csv(filename, collapsed)

def encode(sequence: str) -> int:
    """
    Encodes a DNA sequence (composed of A, G, C, T) to an integer,
    using mapping: A->1, G->2, C->3, T->4.
    """
    mapping = {'A': 1, 'G': 2, 'C': 3, 'T': 4}
    result = 0
    for char in sequence:
        result = result * 10 + mapping[char]
    return result


def decode(number: int) -> str:
    """
    Decodes an integer (encoded via the above encode function) back to a DNA sequence.
    """
    mapping = {'1': 'A', '2': 'G', '3': 'C', '4': 'T'}
    s = str(number)
    return ''.join(mapping[digit] for digit in s)


def compute_global_p(current_n: float, remaining_substrate: float, substrate_capacity_initial: float,
                     s: float, k: float, c: float) -> float:
    """
    Compute the global amplification probability using the provided PCR formula.
    current_N: Total copies in the system.
    remaining_substrate: Remaining substrate capacity.
    substrate_capacity_initial: The initial substrate capacity.
    S: Threshold parameter.
    K: Half-saturation constant (typically S * 10).
    C: Sharpness of phase transition.
    """
    sub_depletion = remaining_substrate / substrate_capacity_initial
    if current_n < s:
        p_equation = k / (k + s)
    else:
        p_equation = (k / (k + current_n)) * ((1 + math.exp(-c * (current_n / (s - 1)))) / 2)
    p = p_equation * sub_depletion
    return p


def process_mutation(sequence: Union[str, int], mutation_rate: float,
                     mutation_probabilities: Dict[str, float]) -> Tuple[Union[str, int], bool]:
    """
    Process mutation over a sequence replication event.
    Each nucleotide is checked with probability `mutation_rate` for mutation.
    If a mutation occurs, one of three types is chosen:
      - substitution: replace nucleotide with a different one.
      - deletion: remove the nucleotide.
      - insertion: insert a new nucleotide before the current one.
    Returns a tuple of (possibly mutated sequence, mutation_occurred flag).
    """
    if isinstance(sequence, int):
        decoded = decode(sequence)
        seq_list = list(decoded)
        mutated = False
        i = 0
        while i < len(seq_list):
            if random.random() < mutation_rate:
                mutated = True
                mutation_type = random.choices(
                    population=['substitution', 'deletion', 'insertion'],
                    weights=[mutation_probabilities['substitution'],
                             mutation_probabilities['deletion'],
                             mutation_probabilities['insertion']],
                    k=1
                )[0]
                if mutation_type == 'substitution':
                    current_nuc = seq_list[i]
                    options = [n for n in NUCLEOTIDES if n != current_nuc]
                    seq_list[i] = random.choice(options)
                    i += 1
                elif mutation_type == 'deletion':
                    del seq_list[i]
                    # Do not increment i; next nucleotide shifts into this position.
                elif mutation_type == 'insertion':
                    inserted = random.choice(NUCLEOTIDES)
                    seq_list.insert(i, inserted)
                    i += 2  # Skip the inserted nucleotide.
            else:
                i += 1
        int_sequence = encode(''.join(seq_list))
        return int_sequence, mutated
    else:
        seq_list = list(sequence)
        mutated = False
        i = 0
        while i < len(seq_list):
            if random.random() < mutation_rate:
                mutated = True
                mutation_type = random.choices(
                    population=['substitution', 'deletion', 'insertion'],
                    weights=[mutation_probabilities['substitution'],
                             mutation_probabilities['deletion'],
                             mutation_probabilities['insertion']],
                    k=1
                )[0]
                if mutation_type == 'substitution':
                    current_nuc = seq_list[i]
                    options = [n for n in NUCLEOTIDES if n != current_nuc]
                    seq_list[i] = random.choice(options)
                    i += 1
                elif mutation_type == 'deletion':
                    del seq_list[i]
                    # Do not increment i; next nucleotide shifts into this position.
                elif mutation_type == 'insertion':
                    inserted = random.choice(NUCLEOTIDES)
                    seq_list.insert(i, inserted)
                    i += 2  # Skip the inserted nucleotide.
            else:
                i += 1
        return ''.join(seq_list), mutated