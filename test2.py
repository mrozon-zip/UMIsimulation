import csv
import ast
import networkx as nx
import matplotlib.pyplot as plt
from Bio import Phylo
from io import StringIO


# --- Data Loading and Conversion ---

def load_data(file_path):
    """
    Load CSV data and convert columns based on a conversion mapping.
    Default mapping:
      "sequence": leave as string
      "n0": convert to int
      "id": convert string to list
      "parent_id": convert to int (empty becomes 0)
    """
    data = []
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # if row doesn't have parent_id assign 0 as a parent_id
            if row[reader.fieldnames[3]] == '':
                row[reader.fieldnames[3]] = 0
            else:
                row[reader.fieldnames[3]] = int(row[reader.fieldnames[3]])
            # ensure that rest of values are in correct format
            row[reader.fieldnames[1]] = int(row[reader.fieldnames[1]])
            row[reader.fieldnames[2]] = ast.literal_eval(row[reader.fieldnames[2]])
            data.append(row)
    return data


def group_data_by_relationship(data):
    """
    Groups dictionaries from `data` into lists of related dictionaries.

    For each dictionary:
      - If its parent_id is 0, it is considered a root.
      - Otherwise, we try to follow its parent chain: a dictionary A is related to dictionary B
        if A['parent_id'] is in B['id'] list.

    Each group will either have a valid root (a dictionary with parent_id == 0)
    or it will be an orphan group (if the chain cannot be resolved).

    Returns:
      groups: a list of groups, where each group is a list of dictionaries.
      orphan_count: number of orphan groups (groups whose key is a missing parent's id).
    """
    # Build a mapping: for each number in any dictionary's "id" list,
    # map that number to the index of the dictionary that contains it.
    mapping = {}
    for idx, d in enumerate(data):
        # Ensure that d['id'] is a list.
        for num in d.get('id', []):
            mapping[num] = idx

    # We'll use memoization to avoid re-computing the root for each dictionary.
    # The memo will map dictionary index to a tuple:
    # ("root", root_index) if a valid chain to a root was found,
    # or ("orphan", missing_parent_value) if the chain breaks.
    memo = {}

    def get_chain_key(i):
        """Iteratively returns a key representing the ultimate grouping for dictionary at index i.
        If a valid root is found, returns ("root", root_index). Otherwise, returns ("orphan", missing_parent).
        Also detects cycles, and groups them with key ("cycle", first_cycle_node)."""
        chain = []
        current = i
        while True:
            if current in memo:
                key = memo[current]
                break
            chain.append(current)
            d = data[current]
            if d.get('parent_id', 0) == 0:
                key = ("root", current)
                break
            parent_val = d.get('parent_id')
            if parent_val not in mapping:
                key = ("orphan", parent_val)
                break
            parent_index = mapping[parent_val]
            if parent_index in chain:
                # Cycle detected; group under the first occurrence
                key = ("cycle", parent_index)
                break
            current = parent_index
        # Memoize the key for all nodes in the chain
        for idx in chain:
            memo[idx] = key
        return key

    # Now group dictionaries by their chain key.
    groups_dict = {}
    for idx in range(len(data)):
        key = get_chain_key(idx)
        groups_dict.setdefault(key, []).append(data[idx])

    # Build final lists and count orphan groups.
    groups = list(groups_dict.values())
    orphan_count = sum(1 for k in groups_dict if k[0] == "orphan")
    return groups, orphan_count


import xml.etree.ElementTree as ET


def is_one_edit_distance(s, t):
    """
    Check if s and t differ by exactly one edit (insertion, deletion, or substitution).
    """
    len_diff = abs(len(s) - len(t))
    if len_diff > 1:
        return False
    # If lengths are equal, check for one substitution.
    if len(s) == len(t):
        diff = sum(1 for a, b in zip(s, t) if a != b)
        return diff == 1
    # When lengths differ by one, check for one insertion/deletion.
    if len(s) > len(t):
        s, t = t, s  # ensure s is the shorter string
    i = j = 0
    found_difference = False
    while i < len(s) and j < len(t):
        if s[i] != t[j]:
            if found_difference:
                return False
            found_difference = True
            j += 1
        else:
            i += 1
            j += 1
    return True


def build_tree_xml(current_index, group, used):
    """
    Recursively builds an XML clade element for the dictionary at current_index.

    Parameters:
      current_index: index of the current dictionary in the group.
      group: list of dictionaries.
      used: set of indices already added to the tree.

    Returns:
      An XML element representing the clade.
    """
    node = group[current_index]
    # Create clade element and add the name (formatted as "sequence,n0")
    clade = ET.Element("clade", branch_length="1")
    name_elem = ET.SubElement(clade, "name")
    name_elem.text = f"{node['sequence']},{node['N0']}"
    used.add(current_index)
    # Look for children: dictionaries not yet used and one edit away from current node's sequence.
    for i, d in enumerate(group):
        if i in used:
            continue
        if is_one_edit_distance(node['sequence'], d['sequence']):
            child_clade = build_tree_xml(i, group, used)
            clade.append(child_clade)
    return clade


def create_phyloxml_for_group(group, output_file):
    """
    Creates a phyloXML file from a group (list of dictionaries).

    The root is chosen as the first dictionary with "parent_id" equal to 0 or "".
    """
    # Find the root dictionary (first one with parent_id == 0 or empty string)
    root_index = None
    for i, d in enumerate(group):
        if d.get("parent_id") in [0, ""]:
            root_index = i
            break
    if root_index is None:
        # If no explicit root is found, use the first dictionary as root.
        root_index = 0

    used = set()
    root_clade = build_tree_xml(root_index, group, used)

    # Build the phyloXML structure.
    phyloxml = ET.Element("phyloxml", xmlns="http://www.phyloxml.org")
    phylogeny = ET.SubElement(phyloxml, "phylogeny", rooted="true")
    # You can change the phylogeny name if needed.
    phy_name = ET.SubElement(phylogeny, "name")
    phy_name.text = "An example"
    phylogeny.append(root_clade)

    # Write the XML tree to the output file.
    tree = ET.ElementTree(phyloxml)
    tree.write(output_file, encoding="UTF-8", xml_declaration=True)


def write_phyloxml_files(list_of_groups):
    """
    Processes a list of groups (each group is a list of dictionaries)
    and writes one phyloXML file per group.
    """
    for idx, group in enumerate(list_of_groups):
        output_file = f"phylo_group{idx + 1}.xml"
        create_phyloxml_for_group(group, output_file)
        print(f"Wrote file: {output_file}")


# Example usage:
if __name__ == "__main__":
    groups, orphan_count = group_data_by_relationship(load_data("results1/polonies_amplified.csv"))
    write_phyloxml_files(groups)
    print("Number of groups:", len(groups))
    # Read a single tree from the file
    tree = Phylo.read('phylo_group1.xml', 'phyloxml')
    Phylo.draw(tree)
    plt.show()

