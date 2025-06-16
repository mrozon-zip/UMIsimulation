import csv
import logging
import matplotlib.pyplot as plt
import networkx as nx
import random
import sys

random.seed(42)

base_folder = "/Users/krzysztofmrozik/Desktop/SciLifeLab/Projects/PCR_simulation/" # For I/O calibration purposes

class Denoiser:
    def __init__(self, input_csv: str):
        self.input_csv = input_csv
        self.data = []
        self.load_data()

    def load_data(self):
        # Load CSV data and convert N0 to integer
        with open(self.input_csv, 'r') as f:
            reader = csv.DictReader(f)
            self.data = [dict(row, N0=(int(row['N0']) if row['N0'] != '' else 0)) for row in reader]
        logging.info(f"Denoiser loaded {len(self.data)} sequences from {self.input_csv}")

    @staticmethod
    def levenshtein(seq1, seq2):
        if len(seq1) < len(seq2):
            return Denoiser.levenshtein(seq2, seq1)

        if len(seq2) == 0:
            return len(seq1)

        previous_row = range(len(seq2) + 1)
        for i, c1 in enumerate(seq1):
            current_row = [i + 1]
            for j, c2 in enumerate(seq2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def collapse_networks(self, output_csv: str):
        """
        Builds a directed network where each node is represented by its 'sequence' and 'N0'.
        A directed edge from node A to node B is added if:
          - The Levenshtein distance between A['sequence'] and B['sequence'] is exactly 1, and
          - N0 of node A >= 2 * (N0 of node B - 1).

        Then, the network is split into weakly connected components.
        Each component is collapsed to its most abundant node (highest N0), with its N0 updated
        to the sum of N0's of all nodes in that component. All nodes in the component get the
        same sequence as the most abundant node.

        The result is written to a CSV file with header: sequence, N0.
        """
        # Build a directed graph
        G = nx.DiGraph()

        # Add nodes from self.data
        for row in self.data:
            sequence = row['sequence']
            N0 = row['N0']
            G.add_node(sequence, N0=N0, sequence=sequence)

        nodes = list(G.nodes())
        # Add directed edges based on Levenshtein distance and abundance condition
        for i, node_a in enumerate(nodes):
            for j, node_b in enumerate(nodes):
                if node_a == node_b:
                    continue
                # Only consider nodes that are 1 edit distance apart
                if self.levenshtein(node_a, node_b) == 1:
                    N0_b = G.nodes[node_b]['N0']
                    threshold = 2 * (N0_b - 1)
                    if G.nodes[node_a]['N0'] >= threshold:
                        G.add_edge(node_a, node_b)

        # Collapse networks based on weakly connected components
        collapsed_results = []
        for component in nx.weakly_connected_components(G):
            total_N0 = 0
            best_node = None
            best_N0 = -1
            for node in component:
                node_N0 = G.nodes[node]['N0']
                total_N0 += node_N0
                if node_N0 > best_N0:
                    best_N0 = node_N0
                    best_node = node
            collapsed_results.append({'sequence': best_node, 'N0': total_N0})

        # Write the collapsed networks to the output CSV file
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['sequence', 'N0'])
            writer.writeheader()
            for row in collapsed_results:
                writer.writerow(row)

        print(f"Collapsed networks written to {output_csv}")

    def directional_networks(self, show=3):
        if not self.data:
            print("Error: Data is not initialized. Please provide a CSV file.")
            return None

        unique_rows = {row['sequence']: row for row in self.data}.values()
        unique_molecules = len(unique_rows)
        print(f"Number of unique molecules before denoising: {unique_molecules}")

        # Initialize the "before" graph
        before_graph = nx.Graph()

        # Add all nodes to the graph
        for row in unique_rows:
            before_graph.add_node(row['sequence'],
                                  sequence=row['sequence'],
                                  amount=int(row['N0']))

        # Add edges based on the condition (this code is left unchanged for reference)
        for row_a in unique_rows:
            for row_b in unique_rows:
                if row_a['sequence'] != row_b['sequence']:
                    value_a = int(row_a['N0'])
                    value_b = int(row_b['N0'])
                    if value_a >= 2 * value_b - 1:
                        before_graph.add_edge(row_a['sequence'], row_b['sequence'])

        print("Graph before filtering edges (value condition only):")
        print(f"Number of nodes: {before_graph.number_of_nodes()}, Number of edges: {before_graph.number_of_edges()}")

        if show in (1, 3):
            nx.draw_spring(before_graph, with_labels=True, node_size=20, font_size=8)
            plt.title("Before Graph")
            plt.show()

        before_graph.remove_edges_from(nx.selfloop_edges(before_graph))

        # Initialize the "after" graph
        after_graph = before_graph.copy()

        for u, v in list(after_graph.edges):
            sequence_u = after_graph.nodes[u]['sequence']
            sequence_v = after_graph.nodes[v]['sequence']
            edit_distance = self.levenshtein(sequence_u, sequence_v)
            if edit_distance >= 2:
                after_graph.remove_edge(u, v)

        print("Graph after filtering edges (edit distance < 2):")
        print(f"Number of nodes: {after_graph.number_of_nodes()}, Number of edges: {after_graph.number_of_edges()}")

        if show in (2, 3):
            nx.draw_spring(after_graph, with_labels=True, node_size=20, font_size=8)
            plt.title("After Graph")
            plt.show()

        return before_graph, after_graph, unique_molecules




if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python denoise.py <input_csv> <output_csv>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]

    denoiser = Denoiser(input_csv)
    # You can test other methods here as needed
    denoiser.collapse_networks(output_csv)
