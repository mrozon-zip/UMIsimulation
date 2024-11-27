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
        sequences = set()

        while len(sequences) < self.number_of_rows:
            sequence = ''.join(random.choices(nucleotides, k=self.length))
            sequences.add(sequence)

        self.df = pd.DataFrame({'Nucleotide Sequence': list(sequences)})
        self.df['Molecule'] = range(1, self.number_of_rows + 1)
        self.df['Type'] = 'original'
        self.df['Edit Distance'] = 0
        self.df['amount'] = 1  # Initialize all molecules with count of 1

        self.df.to_csv(output_filename, index=False)
        print(f"File '{output_filename}' created with {self.number_of_rows} unique rows and sequence length of {self.length}.")

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
                if random.random() < error_rate:
                    error_type_choice = random.choices(
                        ['substitution', 'deletion', 'insertion'],
                        weights=[error_types['substitution'], error_types['deletion'], error_types['insertion']],
                        k=1
                    )[0]

                    if error_type_choice == 'substitution':
                        sequence_list[i] = random.choice([n for n in nucleotides if n != sequence_list[i]])
                        i += 1
                    elif error_type_choice == 'deletion':
                        del sequence_list[i]
                    elif error_type_choice == 'insertion':
                        sequence_list.insert(i + 1, random.choice(nucleotides))
                        i += 2
                else:
                    i += 1

            return ''.join(sequence_list)

        for cycle in range(amplification_cycles):
            new_rows = []
            for _, row in self.df.iterrows():
                sequence = row['Nucleotide Sequence']
                molecule = row['Molecule']
                type_value = row['Type']

                for _ in range(row['amount']):
                    if random.random() < amplification_probability:
                        mutated_sequence = random_errors(sequence, error_rate, error_types)
                        if mutated_sequence != sequence:
                            edit_distance = levenshtein(sequence, mutated_sequence)
                            new_rows.append({
                                'Nucleotide Sequence': mutated_sequence,
                                'Molecule': molecule,
                                'Type': 'error',
                                'Edit Distance': edit_distance,
                                'amount': 1
                            })
                        else:
                            row['amount'] += 1  # Increment amount if no error occurs

            # Update the DataFrame with the new rows and modified "amount"
            self.df = pd.concat([self.df, pd.DataFrame(new_rows)], ignore_index=True)

            # Print sum of the "amount" column
            total_amount = self.df['amount'].sum()
            print(f"Cycle {cycle + 1} complete. Total amount of sequences: {total_amount}")

        self.df.to_csv(output_filename, index=False)
        print(f"File '{output_filename}' created with {len(self.df)} rows.")

    def PCR_analyze(self):
        if self.df is not None:
            total_sequences = self.df['amount'].sum()
            print(f"Total molecules after amplification: {total_sequences}")
        else:
            print("Error: True UMIs have not been created yet.")


class Denoiser:
    def __init__(self, csv_file=None):
        self.df = pd.read_csv(csv_file) if csv_file else None

    def simple(self, threshold):
        if self.df is None:
            print("Error: DataFrame is not initialized. Please provide a CSV file.")
            return None

        sequence_counts = self.df.groupby('Nucleotide Sequence')['amount'].sum()
        valid_sequences = sequence_counts[sequence_counts >= threshold].index

        filtered_df = self.df[self.df['Nucleotide Sequence'].isin(valid_sequences)]
        collapsed_df = filtered_df.groupby(['Nucleotide Sequence', 'Molecule']).agg({
            'Type': 'first',
            'Edit Distance': 'first',
            'amount': 'sum'
        }).reset_index()

        collapsed_df.to_csv('simple_result_enhanced.csv', index=False)
        print(f"Enhanced results saved to 'simple_result_enhanced.csv' with {len(collapsed_df)} rows.")
        return collapsed_df

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
