import os
import csv
import random

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


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python test2.py <input_csv_file>")
    else:
        process_file(sys.argv[1])