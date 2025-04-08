import csv

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