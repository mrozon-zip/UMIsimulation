import pandas as pd
import random
from polyleven import levenshtein
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

random.seed(42)

class simPCR:
    def __init__(self, length, number_of_rows):
        self.length = length
        self.number_of_rows = number_of_rows
        self.df = None  # DataFrame to store the UMIs

    def create_true_UMIs(self, output_filename='true_UMIs.csv'):
        nucleotides = ['A', 'C', 'G', 'T']
        sequences = set()

        while len(sequences) < self.number_of_rows:
            sequence = ''.join(random.choices(nucleotides, k=self.length))
            sequences.add(sequence)

        self.df = pd.DataFrame(list(sequences), columns=['Nucleotide Sequence'])
        self.df['edit distance'] = 0
        self.df['amount'] = 1
        self.df['type'] = 'original'
        self.df['molecule'] = list(range(1, len(self.df) + 1))  # Assign row numbers starting from 1

        self.df.to_csv(output_filename, index=False)
        print(f"File '{output_filename}' created with {self.number_of_rows} unique rows.")

    def save_duplicates_with_revert_and_cross_info(self, df, true_umis_file,
                                                   output_filename='reverting_and_cross_duplicates.csv'):
        # Read the true_UMIs file
        true_umis_df = pd.read_csv(true_umis_file)
        true_umis_dict = true_umis_df.set_index('molecule')['Nucleotide Sequence'].to_dict()

        # Prepare a list to store structured rows
        structured_rows = []

        # Group by 'Nucleotide Sequence' to find duplicates
        grouped = df.groupby('Nucleotide Sequence', sort=False)

        for seq, group in grouped:
            if len(group) > 1:  # Process only groups with duplicates
                group_rows = group.to_dict('records')

                # Log the first occurrence
                original = group_rows[0]
                structured_rows.append({
                    'Description': 'Original',
                    'True UMI Sequence': original['Nucleotide Sequence'],
                    **original
                })

                # Log duplicates with reverting and cross-lineage information
                for idx, duplicate in enumerate(group_rows[1:], start=1):
                    molecule = duplicate['molecule']
                    duplicate_sequence = duplicate['Nucleotide Sequence']
                    duplicate_type = duplicate['type']

                    # Fetch the original sequence for this molecule
                    true_umi_sequence = true_umis_dict.get(molecule, 'N/A')

                    # Determine if it is a revert duplicate
                    if duplicate_sequence == true_umi_sequence and duplicate_type.startswith('error'):
                        description = 'Revert Duplicate 1st Degree'
                    elif any(
                            duplicate_sequence == group_row['Nucleotide Sequence']
                            for group_row in group_rows[:idx]
                    ):
                        description = 'Revert Duplicate 2nd Degree'
                    else:
                        # Check for cross-lineage duplicates
                        cross_lineage = any(
                            duplicate_sequence == original_seq and molecule != other_molecule
                            for other_molecule, original_seq in true_umis_dict.items()
                        )
                        if cross_lineage:
                            description = 'Other Duplicate'
                        else:
                            description = 'Other Duplicate'

                    structured_rows.append({
                        'Description': description,
                        'True UMI Sequence': true_umi_sequence,
                        **duplicate
                    })

        # Create a DataFrame from structured rows
        output_df = pd.DataFrame(structured_rows)

        # Save to a CSV file
        output_df.to_csv(output_filename, index=False)
        print(f"Reverting and cross-lineage duplicates saved to '{output_filename}'.")

    def amplify_with_errors(self, amplification_probability, error_rate, error_types, amplification_cycles,
                            output_filename='amplified_UMIs.csv'):
        df_new = self.df.copy()  # Initialize from the original UMI DataFrame


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
            rows_to_iterate = df_new.copy()  # Work on a copy of the current DataFrame to avoid issues during iteration

            for _, row in rows_to_iterate.iterrows():
                sequence = row['Nucleotide Sequence']
                amount = row['amount']
                row_type = row['type']
                molecule = row['molecule']
                for i in range(amount):
                    if random.random() < amplification_probability:
                        mutated_sequence = random_errors(sequence)
                        if mutated_sequence == sequence:
                            # Increment amount for the existing sequence
                            df_new.loc[df_new['Nucleotide Sequence'] == sequence, 'amount'] += 1
                        else:
                            # Add mutated sequence as a new row in df_new
                            if row_type == 'original':
                                row_type = 'error1'
                            elif row_type.startswith('error'):
                                # Increment the error number
                                error_number = int(row_type[5:]) + 1
                                row_type = f'error{error_number}'
                            df_new = pd.concat([df_new, pd.DataFrame({
                                'Nucleotide Sequence': [mutated_sequence],
                                'edit distance': [0],  # Adjust as needed
                                'amount': [1],
                                'type': row_type,
                                'molecule': molecule
                            })], ignore_index=True)

            print(f"Cycle {cycle + 1}: {len(df_new)} rows")
        print(f"Unique sequences: {df_new['Nucleotide Sequence'].nunique()}")

        # self.save_duplicates_with_revert_and_cross_info(
        #     df_new,
        #     true_umis_file='true_UMIs.csv',
        #     output_filename='filtered_collapsed_duplicates.csv'
        # )
        #
        df_new = (
            df_new.groupby('Nucleotide Sequence', as_index=False, sort=False)  # Disable automatic sorting
            .agg({
                'amount': 'sum',  # Sum the 'amount' values
                'type': 'first',  # Preserve the 'type' of the first occurrence
                'molecule': 'first'  # Preserve the 'molecule' of the first occurrence
            })
        )

        df_new.to_csv(output_filename, index=False)
        print(f"File '{output_filename}' created with {len(df_new)} rows.")

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
        # possibly repeated step since it exists in the amplify with errors.
        unique_rows = self.df.drop_duplicates(subset=['Nucleotide Sequence'])
        unique_molecules = len(unique_rows)
        print(f"Number of unique molecules before denoising: {unique_molecules}")

        # Initialize the "before" graph
        before_graph = nx.Graph()

        # Step 1: Add all nodes to the graph
        for i, row in unique_rows.iterrows():
            before_graph.add_node(f"{row['Nucleotide Sequence']}",
                                  sequence=row['Nucleotide Sequence'],
                                  amount=row['amount'],
                                  molecule=row['molecule'])

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