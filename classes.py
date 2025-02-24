import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import random
import csv
#
random.seed(42)


class SimPcr:
    def __init__(self, length, number_of_rows):
        self.length = length
        self.number_of_rows = number_of_rows
        self.data = []  # List to store UMI dictionaries

    def create_true_umis(self, output_filename='true_UMIs.csv'):
        nucleotides = ['A', 'C', 'G', 'T']
        sequences = set()

        while len(sequences) < self.number_of_rows:
            sequence = ''.join(random.choices(nucleotides, k=self.length))
            sequences.add(sequence)

        self.data = [
            {
                'Sequence': sequence,
                'edit distance': 0,
                'amount': 1,
            }
            for idx, sequence in enumerate(sequences)
        ]

        # Save to a CSV file
        with open(output_filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.data[0].keys())
            writer.writeheader()
            writer.writerows(self.data)

        print(f"File '{output_filename}' created with {self.number_of_rows} unique rows.")

    def amplify_with_errors(self, amplification_probability, error_rate, error_types, amplification_cycles,
                            output_filename='amplified_UMIs.csv'):
        def random_errors(sequence):
            nucleotides = ['A', 'C', 'G', 'T']
            sequence_list = list(sequence)
            i = 0

            while i < len(sequence_list):
                if random.random() < error_rate:
                    error_type = random.choices(
                        ['substitution', 'deletion', 'insertion'],
                        weights=[error_types['substitution'], error_types['deletion'], error_types['insertion']],
                        k=1
                    )[0]

                    if error_type == 'substitution':
                        sequence_list[i] = random.choice([n for n in nucleotides if n != sequence_list[i]])
                        i += 1
                    elif error_type == 'deletion':
                        del sequence_list[i]
                    elif error_type == 'insertion':
                        sequence_list.insert(i + 1, random.choice(nucleotides))
                        i += 2
                else:
                    i += 1

            return ''.join(sequence_list)

        for cycle in range(amplification_cycles):
            new_data = []

            for row in self.data:
                sequence = row['Sequence']
                amount = row['amount']
                row_type = row['type']
                molecule = row['molecule']

                for _ in range(amount):
                    if random.random() < amplification_probability:
                        mutated_sequence = random_errors(sequence)
                        if mutated_sequence == sequence:
                            row['amount'] += 1
                        else:
                            if row_type == 'original':
                                new_type = 'error1'
                            elif row_type.startswith('error'):
                                error_number = int(row_type[5:]) + 1
                                new_type = f'error{error_number}'
                            else:
                                new_type = row_type

                            new_data.append({
                                'Sequence': mutated_sequence,
                                'edit distance': 0,  # Adjust as needed
                                'amount': 1,
                                'type': new_type,
                                'molecule': molecule
                            })

            self.data.extend(new_data)
            print(f"Cycle {cycle + 1}: {len(self.data)} rows")

        # Aggregate the final data
        sequence_map = {}
        for row in self.data:
            seq = row['Sequence']
            if seq not in sequence_map:
                sequence_map[seq] = row.copy()
            else:
                sequence_map[seq]['amount'] += row['amount']

        self.data = list(sequence_map.values())

        # Save the amplified UMIs to a CSV file
        with open(output_filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.data[0].keys())
            writer.writeheader()
            writer.writerows(self.data)

        print(f"File '{output_filename}' created with {len(self.data)} rows.")


class Denoiser:
    def __init__(self, csv_file=None):
        self.data = []  # Initialize as an empty list to hold rows as dictionaries
        if csv_file:
            with open(csv_file, mode='r') as file:
                reader = csv.DictReader(file)
                self.data = [row for row in reader]

    def simple(self, threshold):
        if not self.data:
            print("Error: Data is not initialized. Please provide a CSV file.")
            return None

        # Filter the data based on the threshold
        valid_sequences = [
            row for row in self.data if int(row['amount']) >= threshold
        ]

        # Save the filtered data to a CSV file
        with open('simple_result.csv', mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=valid_sequences[0].keys())
            writer.writeheader()
            writer.writerows(valid_sequences)

        print(f"Enhanced results saved to 'simple_result.csv' with {len(valid_sequences)} rows.")

        return valid_sequences

    def directional_networks(self, show=3):
        if not self.data:
            print("Error: Data is not initializded. Please provide a CSV file.")
            return None

        unique_rows = {row['Sequence']: row for row in self.data}.values()
        unique_molecules = len(unique_rows)
        print(f"Number of unique molecules before denoising: {unique_molecules}")

        # Initialize the "before" graph
        before_graph = nx.Graph()

        # Add all nodes to the graph
        for row in unique_rows:
            before_graph.add_node(row['Sequence'],
                                  sequence=row['Sequence'],
                                  amount=int(row['amount']),
                                  molecule=row['molecule'])

        # Add edges based on the condition
        for row_a in unique_rows:
            for row_b in unique_rows:
                if row_a['Sequence'] != row_b['Sequence']:
                    value_a = int(row_a['amount'])
                    value_b = int(row_b['amount'])

                    if value_a >= 2 * value_b - 1:
                        before_graph.add_edge(row_a['Sequence'], row_b['Sequence'])

        print("Graph before filtering edges (value condition only):")
        print(f"Number of nodes: {before_graph.number_of_nodes()}, Number of edges: {before_graph.number_of_edges()}")

        # Show the "before" graph if requested
        if show in (1, 3):
            nx.draw_spring(before_graph, with_labels=True, node_size=20, font_size=8)
            plt.title("Before Graph")
            plt.show()

        before_graph.remove_edges_from(nx.selfloop_edges(before_graph))

        # Initialize the "after" graph
        after_graph = before_graph.copy()

        # Remove edges from the "after" graph based on edit distance condition
        for u, v in list(after_graph.edges):
            sequence_u = after_graph.nodes[u]['sequence']
            sequence_v = after_graph.nodes[v]['sequence']
            edit_distance = self.levenshtein(sequence_u, sequence_v)
            if edit_distance >= 2:
                after_graph.remove_edge(u, v)

        print("Graph after filtering edges (edit distance < 2):")
        print(f"Number of nodes: {after_graph.number_of_nodes()}, Number of edges: {after_graph.number_of_edges()}")

        # Show the "after" graph if requested
        if show in (2, 3):
            nx.draw_spring(after_graph, with_labels=True, node_size=20, font_size=8)
            plt.title("After Graph")
            plt.show()

        return before_graph, after_graph, unique_molecules

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

    @staticmethod
    def networks_resolver(graph, toggle="central_node"):
        """
        Resolve networks in different ways based on the toggle parameter.

        Parameters:
        - graph: The NetworkX graph to analyze.
        - toggle: The resolution method to use (default: "central_node").

        Returns:
        - A list of dictionaries containing resolved network data.
        """
        central_nodes_data = []

        if toggle == "central_node":
            # Resolve networks using the central node approach
            for component in nx.connected_components(graph):
                subgraph = graph.subgraph(component)
                central_node = max(
                    subgraph.nodes(data=True), key=lambda x: x[1].get('amount', 0)
                )
                central_node_id = central_node[0]
                central_node_amount = central_node[1].get('amount', 0)
                sequence = central_node[1].get('sequence', '')

                # Calculate the total amount of all connected nodes
                total_amount = central_node_amount
                for node in subgraph.nodes(data=True):
                    total_amount += node[1].get('amount', 0)

                # Add the resolved data to the list
                central_nodes_data.append({
                    'Sequence': sequence,
                    'Central Node Count': central_node_amount,
                    'Network Nodes Count': total_amount
                })

            # Save the results to a CSV file
            with open('directional_results.csv', mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=central_nodes_data[0].keys())
                writer.writeheader()
                writer.writerows(central_nodes_data)

            print("Central nodes saved to 'directional_results.csv'.")

        else:
            print(f"Toggle '{toggle}' is not implemented.")
            return None

        return central_nodes_data

    def analysis(self, denoised_file, true_umis_file, amplified_umis_file="amplified_UMIs.csv"):
        def load_file(file):
            with open(file, mode='r') as f:
                return {row['Sequence'] for row in csv.DictReader(f)}

        sequences_denoised = load_file(denoised_file)
        sequences_true = load_file(true_umis_file)
        sequences_amplified = load_file(amplified_umis_file)

        tp = len(sequences_denoised & sequences_true)
        fn = len(sequences_true - sequences_denoised)
        fp = len(sequences_denoised - sequences_true)
        tn = len(sequences_amplified - sequences_denoised - sequences_true)

        cm = [[tp, fp], [fn, tn]]

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Actual Positives", "Actual Negatives"],
                    yticklabels=["Predicted Positives", "Predicted Negatives"])
        plt.title("Confusion Matrix")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.show()

        print(f"True Positives (TP): {tp}")
        print(f"True Negatives (TN): {tn}")
        print(f"False Positives (FP): {fp}")
        print(f"False Negatives (FN): {fn}")

        return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}

    @staticmethod
    def node_probe(graph, tier1=5, tier2=3):
        """
        Draws a subgraph centered on a randomly selected node with at least `tier1` connections.
        The subgraph includes tier-1 connections and up to `tier2` random tier-2 connections.

        Parameters:
        - graph: The input graph (NetworkX object) to probe.
        - tier1: Minimum number of connections required for a node to be considered (default: 5).
        - tier2: Maximum number of tier-2 connections to display for each tier-1 node (default: 3).

        Returns:
        - None. Displays the subgraph visualization.
        """
        # Find nodes with at least `tier1` connections
        eligible_nodes = [node for node in graph.nodes if len(list(graph.neighbors(node))) >= tier1]

        if not eligible_nodes:
            print(f"No nodes found with at least {tier1} connections.")
            return

        # Randomly select a node from eligible nodes
        selected_node = random.choice(eligible_nodes)
        print(f"Selected node: {selected_node}")

        # Get tier-1 connections
        tier1_nodes = list(graph.neighbors(selected_node))

        # Build the subgraph
        subgraph_nodes = {selected_node}
        subgraph_nodes.update(tier1_nodes)  # Add tier-1 nodes

        for tier1_node in tier1_nodes:
            tier2_neighbors = list(graph.neighbors(tier1_node))
            # Select up to `tier2` random tier-2 neighbors
            random_tier2 = random.sample(tier2_neighbors, min(len(tier2_neighbors), tier2))
            subgraph_nodes.update(random_tier2)

        # Create the subgraph
        subgraph = graph.subgraph(subgraph_nodes)

        # Visualize the subgraph
        print(f"Visualizing subgraph with {len(subgraph.nodes)} nodes and {len(subgraph.edges)} edges.")
        nx.draw_spring(subgraph, with_labels=True, node_size=500, node_color="lightblue", font_size=10)
        plt.title("Node Probe Subgraph")
        plt.show()
