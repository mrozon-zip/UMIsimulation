import csv
import networkx as nx
import matplotlib.pyplot as plt


def read_csv_file(filepath):
    with open(filepath, 'r', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def find_self_parent_nodes(data):
    self_parent_nodes = [row for row in data if row['id'] == row['parent_id']]
    return self_parent_nodes

# Load data
result = read_csv_file('results1/polonies_amplified.csv')

# Find and count self-parent nodes
self_parent_nodes = find_self_parent_nodes(result)

# Output the count and the entries
print(f"Count of nodes where 'id' == 'parent_id': {len(self_parent_nodes)}")
print("Matching entries:")
for node in self_parent_nodes:
    print(node)