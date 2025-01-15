import pandas as pd
import random
from polyleven import levenshtein
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns


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

        # Initialize the 'root', 'Type', and 'Edit Distance' columns
        self.df['root'] = self.df['Molecule'].astype(str)  # Copy Molecule to root
        self.df['Type'] = 'original'  # All initial sequences are original
        self.df['Edit Distance'] = 0  # Initialize the Edit Distance column

        # Reorder the columns to match the desired output
        self.df = self.df[['root', 'Molecule', 'Nucleotide Sequence', 'Type', 'Edit Distance']]

        # Output the resulting DataFrame to a CSV file
        self.df.to_csv(output_filename, index=False)
        print(
            f"File '{output_filename}' has been created with {self.number_of_rows} unique rows and sequence length of {self.length}.")

    def amplify_with_errors(self, amplification_probability, error_rate, error_types, amplification_cycles,
                            output_filename='amplified_UMIs.csv'):
        if self.df is None:
            print("Error: True UMIs have not been created yet.")
            return

        def random_errors(sequence, error_rate, error_types):
            nucleotides = ['A', 'C', 'G', 'T']
            sequence_list = list(sequence)
            i = 0

            while i < len(sequence_list):
                if random.random() < error_rate:  # Introduce an error
                    error_type_choice = random.choices(
                        ['substitution', 'deletion', 'insertion'],
                        weights=[error_types['substitution'], error_types['deletion'], error_types['insertion']],
                        k=1
                    )[0]

                    if error_type_choice == 'substitution':
                        possible_mutations = [n for n in nucleotides if n != sequence_list[i]]
                        sequence_list[i] = random.choice(possible_mutations)
                        i += 1  # Move to the next nucleotide

                    elif error_type_choice == 'deletion':
                        del sequence_list[i]  # Remove the current nucleotide
                        # Note: Do not increment `i` to check the next nucleotide in place of the deleted one

                    elif error_type_choice == 'insertion':
                        new_nucleotide = random.choice(nucleotides)
                        sequence_list.insert(i + 1, new_nucleotide)  # Insert after the current nucleotide
                        i += 2  # Skip the newly inserted nucleotide and move to the next nucleotide

                else:
                    i += 1  # Move to the next nucleotide if no error occurs

            return ''.join(sequence_list)

        def get_next_molecule_number(df):
            """Find the next available unique molecule number, incremental to the highest existing number."""
            max_molecule_number = df['Molecule'].max()
            return max_molecule_number + 1

        # Perform amplification for the specified number of cycles
        for cycle in range(amplification_cycles):
            amplified_rows = []
            next_molecule_number = get_next_molecule_number(self.df)

            # Iterate through each row and check for amplification and errors
            for index, row in self.df.iterrows():
                sequence = row['Nucleotide Sequence']
                # molecule_number = row['Molecule']
                type_value = row['Type']
                root_value = row['root']  # Get the root value

                # Append the original row (no changes)
                amplified_rows.append(row)

                # If the row is amplified
                if random.random() < amplification_probability:
                    row_copy = row.copy()

                    # Apply polymerase errors with a probability
                    mutated_sequence = random_errors(sequence, error_rate, error_types)
                    row_copy['Nucleotide Sequence'] = mutated_sequence

                    # Change the 'root' value for the amplified row
                    row_copy['root'] = f"{root_value}a"  # Add 'a' to the root value

                    # Check if the sequence was actually mutated
                    if mutated_sequence != sequence:
                        # Assign error type and compute new edit distance
                        error_number = int(type_value[5:]) + 1 if type_value.startswith('error') else 1
                        row_copy['Type'] = f'error{error_number}'

                        # Calculate new edit distance: existing + distance from original to mutated
                        edit_distance = levenshtein(sequence, mutated_sequence)
                        row_copy['Edit Distance'] = edit_distance
                        row_copy['Molecule'] = next_molecule_number  # Update molecule number
                        next_molecule_number += 1  # Ensure the next unique number is used
                    else:
                        # If no error occurs during amplification, retain existing distance
                        row_copy['Type'] = type_value  # Maintain original type
                        row_copy['Edit Distance'] = 0  # No error, so distance is 0

                    # Add the modified row (amplified and possibly mutated)
                    amplified_rows.append(row_copy)

            # At the end of each cycle, calculate edit distance for all rows
            for amplified_row in amplified_rows:
                root_stripped = amplified_row['root'].rstrip('a')  # Strip the 'a' for comparison
                corresponding_row = self.df[self.df['root'] == root_stripped]

                if not corresponding_row.empty:
                    corresponding_sequence = corresponding_row.iloc[0]['Nucleotide Sequence']
                    edit_distance = levenshtein(amplified_row['Nucleotide Sequence'], corresponding_sequence)
                    amplified_row['Edit Distance'] = edit_distance

            # Update the DataFrame for the next cycle
            self.df = pd.DataFrame(amplified_rows, columns=self.df.columns)
            print(f"Cycle {cycle + 1} complete. The DataFrame now has {len(self.df)} rows.")

        # Save the resulting amplified DataFrame to a CSV file
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
        if csv_file:
            self.df = pd.read_csv(csv_file)  # Load DataFrame from CSV if provided
        else:
            self.df = None  # Initialize an empty DataFrame

    def simple(self, threshold):
        if self.df is None:
            print("Error: DataFrame is not initialized. Please provide a CSV file.")
            return None

        # Count appearances of each sequence
        sequence_counts = self.df['Nucleotide Sequence'].value_counts()

        # Fetch all sequences that appear 'threshold' times or more
        valid_sequences = sequence_counts[sequence_counts >= threshold].index

        # Filter the DataFrame to keep only rows with valid sequences
        filtered_df = self.df[self.df['Nucleotide Sequence'].isin(valid_sequences)]

        # Collapse rows into one per unique molecule, aggregating the other columns
        collapsed_df = filtered_df.groupby(['Molecule']).agg({
            'Nucleotide Sequence': 'first',  # Choose the first occurrence
            'Type': 'first',  # Choose the first occurrence
            'Edit Distance': 'first'
        }).reset_index()

        # Output the collapsed DataFrame to 'simple_denoiser_result_enhanced.csv'
        collapsed_df.to_csv('simple_result_enhanced.csv', index=False)
        print(f"Enhanced results saved to 'simple_result_enhanced.csv' with {len(collapsed_df)} rows.")

        # Create another DataFrame for the simple result with a new 'Molecule' column
        simple_result_df = collapsed_df.copy()
        simple_result_df['Molecule'] = range(1, len(simple_result_df) + 1)  # Generate new molecule numbers

        # Output the simple result to 'simple_denoiser_result.csv'
        simple_result_df[['Molecule', 'Nucleotide Sequence']].to_csv('simple_result.csv', index=False)
        print(f"Simple results saved to 'simple_result.csv' with {len(simple_result_df)} rows.")

        return collapsed_df.reset_index(drop=True)  # Return the collapsed DataFrame

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

        Returns:#
        - before_graph: The graph before edge filtering by edit distance.
        - after_graph: The graph after edge filtering by edit distance.
        - unique_molecules: Number of unique molecules in the DataFrame.
        """
        if self.df is None:
            print("Error: DataFrame is not initialized. Please provide a CSV file.")
            return None

        # Count occurrences of each Molecule
        count_df = self.df['Molecule'].value_counts().reset_index()
        count_df.columns = ['Molecule', 'amount']

        # Merge the count back into the original DataFrame
        merged_df = pd.merge(self.df, count_df, on='Molecule')

        # Keep only the first occurrence of each Molecule
        result_df = merged_df.drop_duplicates(subset=['Molecule'])
        unique_molecules = len(result_df)
        print(f"Number of unique molecules before denoising: {unique_molecules}")

        # Initialize the "before" graph
        before_graph = nx.Graph()

        # Add nodes and edges to the "before" graph
        for i, row_a in result_df.iterrows():
            before_graph.add_node(f"{row_a['Nucleotide Sequence']}",
                                  sequence=row_a['Nucleotide Sequence'],
                                  amount=row_a['amount'],
                                  molecule=row_a['Molecule'])
            for j, row_b in result_df.iterrows():
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
                    'Central node ID': central_node_id,
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

        # Extract sequences from the detected columns
        sequences_denoised = set(denoised_results[sequence_col_denoised])
        sequences_true = set(true_umis[sequence_col_true])
        sequences_amplified = set(amplified_umis[sequence_col_amplified])

        # Calculate confusion matrix components
        tp = len(sequences_denoised.intersection(sequences_true))  # True Positives
        tn = len(sequences_amplified - sequences_denoised - sequences_true)  # True Negatives
        fp = len(sequences_denoised - sequences_true)  # False Positives
        fn = len(sequences_true - sequences_denoised)  # False Negatives

        # Create the confusion matrix
        cm = [[tp, fn], [fp, tn]]

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