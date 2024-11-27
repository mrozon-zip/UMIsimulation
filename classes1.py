import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
from polyleven import levenshtein

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
        if self.df is None:
            print("Error: DataFrame is not initialized.")
            return None

        count_df = self.df.groupby('Molecule')['amount'].sum().reset_index()
        merged_df = pd.merge(self.df, count_df, on='Molecule')

        before_graph = nx.Graph()
        for _, row in merged_df.iterrows():
            before_graph.add_node(row['Nucleotide Sequence'], sequence=row['Nucleotide Sequence'], amount=row['amount'])
            for _, other_row in merged_df.iterrows():
                if row['Nucleotide Sequence'] != other_row['Nucleotide Sequence']:
                    if row['amount'] >= 2 * other_row['amount'] - 1:
                        before_graph.add_edge(row['Nucleotide Sequence'], other_row['Nucleotide Sequence'])

        before_graph.remove_edges_from(nx.selfloop_edges(before_graph))
        if show in (1, 3):
            nx.draw_spring(before_graph, with_labels=True, node_size=20, font_size=8)
            plt.title("Before Graph")
            plt.show()

        after_graph = before_graph.copy()
        for u, v in list(after_graph.edges):
            if levenshtein(u, v) >= 2:
                after_graph.remove_edge(u, v)

        after_graph.remove_edges_from(nx.selfloop_edges(after_graph))
        if show in (2, 3):
            nx.draw_spring(after_graph, with_labels=True, node_size=20, font_size=8)
            plt.title("After Graph")
            plt.show()

        return before_graph, after_graph

    def analysis(self, denoised_file, true_umis_file, amplified_umis_file="amplified_UMIs.csv"):
        def detect_sequence_column(df):
            for col in df.columns:
                if df[col].dropna().apply(lambda x: isinstance(x, str) and all(c in 'ATCG' for c in x)).all():
                    return col
            raise ValueError("No valid sequence column found in the DataFrame.")

        denoised_results = pd.read_csv(denoised_file)
        true_umis = pd.read_csv(true_umis_file)
        amplified_umis = pd.read_csv(amplified_umis_file)

        seq_col_denoised = detect_sequence_column(denoised_results)
        seq_col_true = detect_sequence_column(true_umis)
        seq_col_amplified = detect_sequence_column(amplified_umis)

        seq_denoised = set(denoised_results[seq_col_denoised])
        seq_true = set(true_umis[seq_col_true])
        seq_amplified = set(amplified_umis[seq_col_amplified])

        tp = len(seq_denoised.intersection(seq_true))
        tn = len(seq_amplified - seq_denoised - seq_true)
        fp = len(seq_denoised - seq_true)
        fn = len(seq_true - seq_denoised)

        cm = [[tp, fn], [fp, tn]]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Actual Positives", "Actual Negatives"],
                    yticklabels=["Predicted Positives", "Predicted Negatives"])
        plt.title("Confusion Matrix")
        plt.show()

        print(f"True Positives: {tp}, False Positives: {fp}, False Negatives: {fn}, True Negatives: {tn}")
        return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}
