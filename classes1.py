import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
from polyleven import levenshtein

random.seed(42)
#
class simPCR:
    def __init__(self, length, number_of_rows):
        self.length = length
        self.number_of_rows = number_of_rows
        self.df = None  # DataFrame to store the UMIs

    def create_true_UMIs(self, output_filename='true_UMIs.csv'):
        # Generate random nucleotide sequences
        nucleotides = ['A', 'C', 'G', 'T']
        sequences = set()  # Use a set to ensure uniqueness

        while len(sequences) < self.number_of_rows:
            sequence = ''.join(random.choices(nucleotides, k=self.length))
            sequences.add(sequence)  # Add sequence to the set

        # Convert the set to a list for DataFrame creation
        sequences = list(sequences)

        # Create a DataFrame with sequences
        self.df = pd.DataFrame(sequences, columns=['Nucleotide Sequence'])

        # Assign molecule numbers from 1 to the number_of_rows to each sequence
        self.df['Molecule'] = range(1, self.number_of_rows + 1)

        # Initialize the 'root', 'Type', 'Edit Distance', and 'amount' columns
        self.df['root'] = self.df['Molecule'].astype(str)
        self.df['Type'] = 'original'
        self.df['Edit Distance'] = 0
        self.df['amount'] = 1

        # Save the resulting DataFrame
        self.df.to_csv(output_filename, index=False)
        print(f"File '{output_filename}' has been created with {self.number_of_rows} unique rows and sequence length of {self.length}.")

    def amplify_with_errors(self, amplification_probability, error_rate, error_types, amplification_cycles, output_filename='amplified_UMIs.csv'):
        if self.df is None:
            print("Error: True UMIs have not been created yet.")
            return

        def introduce_errors(sequence, error_rate, error_types):
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
                        sequence_list.insert(i, random.choice(nucleotides))
                        i += 1
                else:
                    i += 1

            return ''.join(sequence_list)

        for cycle in range(amplification_cycles):
            original_df = self.df.copy()
            new_rows = []

            for index, row in original_df.iterrows():
                if random.random() > amplification_probability:
                    continue

                initial_amount = row['amount']
                new_amount = initial_amount

                for _ in range(initial_amount):
                    sequence = row['Nucleotide Sequence']
                    mutated_sequence = introduce_errors(sequence, error_rate, error_types)
                    changes_made = mutated_sequence != sequence

                    if changes_made:
                        new_rows.append({
                            "root": row['Molecule'],
                            "Molecule": self.df['Molecule'].max() + 1,
                            "Nucleotide Sequence": mutated_sequence,
                            "Type": 'error' if row['Type'] == 'original' else row['Type'],
                            "Edit Distance": levenshtein(sequence, mutated_sequence),
                            "amount": initial_amount
                        })
                    else:
                        new_amount += 1

                self.df.at[index, 'amount'] = new_amount

            if new_rows:
                self.df = pd.concat([self.df, pd.DataFrame(new_rows)], ignore_index=True)

            print(f"Cycle {cycle + 1} complete. DataFrame now has {len(self.df)} rows.")
        self.df['Molecule'] = range(1, len(self.df) + 1)
        self.df.to_csv(output_filename, index=False)
        print(f"File '{output_filename}' has been created with {len(self.df)} rows.")

    def PCR_analyze(self):
        if self.df is not None:
            unique_molecule_numbers = self.df['Molecule'].nunique()
            print(f"Number of unique Molecule numbers: {unique_molecule_numbers}")
        else:
            print("Error: True UMIs have not been created yet.")


class Denoiser:
    def __init__(self, csv_file=None):
        self.df = pd.read_csv(csv_file) if csv_file else None

    def simple(self, threshold):
        if self.df is None:
            print("Error: DataFrame is not initialized. Please provide a CSV file.")
            return None

        # Filter the DataFrame based on the threshold
        valid_sequences_df = self.df.loc[self.df['amount'] >= threshold]

        # Save the filtered DataFrame to a CSV file
        valid_sequences_df.to_csv('simple_result.csv', index=False)
        print(f"Enhanced results saved to 'simple_result.csv' with {len(valid_sequences_df)} rows.")

        # Return the filtered DataFrame
        return valid_sequences_df

    def directional_networks(self, show=3):
        """
        Creates and optionally visualizes two network graphs:
        - Before filtering edges by edit distance ("before" graph).
        - After filtering edges by edit distance ("after" graph).

        Parameters:
        - show (int): Controls which graphs to display.
          0 - Show no graphs.
          1 - Show "before" graph only.
          2 - Show "after" graph only.
          3 - Show both graphs (default).

        Returns:
        - before_graph: The graph before edge filtering by edit distance.
        - after_graph: The graph after edge filtering by edit distance.
        - unique_molecules: Number of unique molecules in the DataFrame.
        """
        if self.df is None:
            print("Error: DataFrame is not initialized. Please provide a CSV file.")
            return None

        # Ensure all rows are considered for nodes
        unique_rows = self.df.drop_duplicates(subset=['Molecule'])
        unique_molecules = len(unique_rows)
        print(f"Number of unique molecules before denoising: {unique_molecules}")

        # Initialize the "before" graph
        before_graph = nx.Graph()

        # Step 1: Add all nodes to the graph
        for i, row in unique_rows.iterrows():
            before_graph.add_node(f"{row['Nucleotide Sequence']}",
                                  sequence=row['Nucleotide Sequence'],
                                  amount=row['amount'],
                                  molecule=row['Molecule'])

        # Step 2: Add edges based on the condition
        for i, row_a in unique_rows.iterrows():
            for j, row_b in unique_rows.iterrows():
                if i != j:  # Prevent self-loops
                    value_a = row_a['amount']
                    value_b = row_b['amount']

                    # Add edge based on value condition
                    if value_a >= 2 * value_b - 1:
                        before_graph.add_edge(f"{row_a['Nucleotide Sequence']}", f"{row_b['Nucleotide Sequence']}")

        print("Graph before filtering edges (value condition only):")
        print(f"Number of nodes: {before_graph.number_of_nodes()}, Number of edges: {before_graph.number_of_edges()}")

        # Show the "before" graph if requested
        if show in (1, 3):
            print("Visualizing 'before' graph...")
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
            edit_distance = levenshtein(sequence_u, sequence_v)
            if edit_distance >= 2:
                after_graph.remove_edge(u, v)

        print("Graph after filtering edges (edit distance < 2):")
        print(f"Number of nodes: {after_graph.number_of_nodes()}, Number of edges: {after_graph.number_of_edges()}")

        # Show the "after" graph if requested
        if show in (2, 3):
            print("Visualizing 'after' graph...")
            nx.draw_spring(after_graph, with_labels=True, node_size=20, font_size=8)
            plt.title("After Graph")
            plt.show()

        return before_graph, after_graph, unique_molecules

    def networks_resolver(self, graph, toggle="central_node"):
        """
        Resolve networks in different ways based on the toggle parameter.

        Parameters:
        - graph: The NetworkX graph to analyze.
        - toggle: The resolution method to use (default: "central_node").
        """
        central_nodes_data = []

        # Toggle for selecting resolution method
        if toggle == "central_node":
            # Analyze central nodes within the given graph
            for component in nx.connected_components(graph):
                subgraph = graph.subgraph(component)
                central_node = max(subgraph.nodes(data=True), key=lambda x: x[1]['amount'])
                central_node_id = central_node[0]
                # central_node_molecule = central_node[1]['molecule']
                central_node_amount = central_node[1]['amount']
                sequence = central_node[1]['sequence']

                # Sum amounts of all nodes connected to the central node
                total_amount = central_node_amount  # Start with the central node's amount
                connected_nodes = nx.descendants(subgraph, central_node_id)

                for node in connected_nodes:
                    total_amount += subgraph.nodes[node]['amount']

                # Append the result
                central_nodes_data.append({
                    'Sequence': sequence,
                    'Central node count': central_node_amount,
                    'Network nodes count': total_amount
                })

            # Create a DataFrame for central nodes data and save to CSV
            central_nodes_df = pd.DataFrame(central_nodes_data)
            central_nodes_df.to_csv('directional_results.csv', index=False)
            print("Central nodes saved to 'directional_result.csv'")

            return central_nodes_df  # Return the central nodes DataFrame
        else:
            print(f"Toggle '{toggle}' is not implemented.")
            return None

    def analysis(self, denoised_file, true_umis_file, amplified_umis_file="amplified_UMIs.csv"):
        """
        Analyze the overlap between denoised_results and true_umis and produce a confusion matrix.

        Parameters:
        - denoised_file: File path or DataFrame containing denoised results.
        - true_umis_file: File path or DataFrame containing true UMIs.
        - amplified_umis_file: File path to the file containing all amplified UMIs (default: "amplified_UMIs.csv").

        Returns:
        - A dictionary with TP, TN, FP, FN counts and a visualization of the confusion matrix.
        """

        def detect_sequence_column(df):
            """Detect the column containing sequences with only 'A', 'T', 'C', and 'G'."""
            for col in df.columns:
                if df[col].dropna().apply(lambda x: isinstance(x, str) and all(c in 'ATCG' for c in x)).all():
                    return col
            raise ValueError("No valid sequence column found in the DataFrame.")

        # Load the inputs if they are file paths
        if isinstance(denoised_file, str):
            denoised_results = pd.read_csv(denoised_file)
        else:
            denoised_results = denoised_file

        if isinstance(true_umis_file, str):
            true_umis = pd.read_csv(true_umis_file)
        else:
            true_umis = true_umis_file

        amplified_umis = pd.read_csv(amplified_umis_file)
        # Detect sequence columns in all DataFrames
        sequence_col_denoised = detect_sequence_column(denoised_results)
        sequence_col_true = detect_sequence_column(true_umis)
        sequence_col_amplified = detect_sequence_column(amplified_umis)

        print(sequence_col_true, sequence_col_denoised, sequence_col_amplified)

        # Extract sequences from the detected columns
        sequences_denoised = set(denoised_results[sequence_col_denoised])
        sequences_true = set(true_umis[sequence_col_true])
        sequences_amplified = set(amplified_umis[sequence_col_amplified])

        # Calculate sets for confusion matrix components
        true_positives = sequences_denoised.intersection(sequences_true)  # In denoised results and true UMIs
        false_negatives = sequences_true - sequences_denoised  # In true UMIs but not in denoised results
        false_positives = sequences_denoised - sequences_true  # In denoised results but not in true UMIs
        true_negatives = sequences_amplified - sequences_denoised - sequences_true  # Not in denoised or true UMIs but in amplified UMIs

        # Calculate counts
        tp = len(true_positives)
        fn = len(false_negatives)
        fp = len(false_positives)
        tn = len(true_negatives)

        # Create the confusion matrix
        cm = [[tp, fp], [fn, tn]]

        # Visualize the confusion matrix
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Actual Positives", "Actual Negatives"],
                    yticklabels=["Predicted Positives", "Predicted Negatives"])
        plt.title("Confusion Matrix")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.show()

        # Display results
        print(f"True Positives (TP): {tp}")
        print(f"True Negatives (TN): {tn}")
        print(f"False Positives (FP): {fp}")
        print(f"False Negatives (FN): {fn}")

        # Return the confusion matrix components
        return {
            "True Positives": tp,
            "True Negatives": tn,
            "False Positives": fp,
            "False Negatives": fn
        }

    def node_probe(self, graph, tier1=5, tier2=3):
        """
        Draws a subgraph centered on a randomly selected node with at least `tier1` connections.
        The subgraph includes tier-1 connections and up to `tier2` random tier-2 connections.

        Parameters:
        - graph: The input graph (NetworkX object) to probe.
        - tier1: Minimum number of connections required for a node to be considered (default: 5).
        - tier2: Maximum number of tier-2 connections to display for each tier-1 node (default: 3).
        """
        # Filter nodes with at least `tier1` connections
        eligible_nodes = [node for node in graph.nodes if len(list(graph.neighbors(node))) >= tier1]

        if not eligible_nodes:
            print(f"No nodes found with at least {tier1} connections.")
            return

        # Randomly select a node from the eligible nodes
        selected_node = random.choice(eligible_nodes)
        print(f"Selected node: {selected_node}")

        # Get tier-1 connections
        tier1_nodes = list(graph.neighbors(selected_node))

        # Build the subgraph
        subgraph_nodes = {selected_node}  # Include the selected node
        subgraph_nodes.update(tier1_nodes)  # Include all tier-1 nodes

        for tier1_node in tier1_nodes:
            tier2_neighbors = list(graph.neighbors(tier1_node))
            # Randomly select up to `tier2` neighbors from tier-2 connections
            random_tier2 = random.sample(tier2_neighbors, min(len(tier2_neighbors), tier2))
            subgraph_nodes.update(random_tier2)

        # Create the subgraph
        subgraph = graph.subgraph(subgraph_nodes)

        # Visualize the subgraph
        print(f"Visualizing subgraph with {len(subgraph.nodes)} nodes and {len(subgraph.edges)} edges.")
        nx.draw_spring(subgraph, with_labels=True, node_size=500, node_color="lightblue", font_size=10)
        plt.show()
