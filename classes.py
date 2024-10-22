import pandas as pd
import random

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

        # Assign random genetic loci between 1 and 8 to each sequence
        genetic_loci = [random.randint(1, 8) for _ in range(self.number_of_rows)]

        # Create a DataFrame with sequences and their corresponding genetic loci
        self.df = pd.DataFrame(sequences, columns=['Nucleotide Sequence'])
        self.df['Genetic Loci'] = genetic_loci

        # Sort the DataFrame by 'Genetic Loci' in increasing order
        self.df.sort_values(by='Genetic Loci', inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        # Assign order numbers from 1 to the number_of_rows to each sequence
        self.df['Order Number'] = range(1, self.number_of_rows + 1)

        # Reorder the columns to match the desired output
        self.df = self.df[['Order Number', 'Nucleotide Sequence', 'Genetic Loci']]

        # Output the resulting DataFrame to a CSV file
        self.df.to_csv(output_filename, index=False)
        print(f"File '{output_filename}' has been created with {self.number_of_rows} rows and sequence length of {self.length}.")

    def amplify_with_errors(self, amplification_probability, error_rate, amplification_cycles, output_filename='amplified_UMIs.csv'):
        if self.df is None:
            print("Error: True UMIs have not been created yet.")
            return

        def random_errors_polymerase(sequence, error_rate):
            nucleotides = ['A', 'C', 'G', 'T']
            sequence_list = list(sequence)

            for i, nucleotide in enumerate(sequence_list):
                if random.random() < error_rate:  # Introduce an error
                    possible_mutations = [n for n in nucleotides if n != nucleotide]
                    sequence_list[i] = random.choice(possible_mutations)

            return ''.join(sequence_list)

        def get_next_order_number(df):
            """Find the next available unique order number, incremental to the highest existing number."""
            max_order_number = df['Order Number'].max()
            return max_order_number + 1

        # Perform amplification for the specified number of cycles
        for cycle in range(amplification_cycles):
            amplified_rows = []
            next_order_number = get_next_order_number(self.df)

            # Iterate through each row and check for amplification and errors
            for index, row in self.df.iterrows():
                sequence = row['Nucleotide Sequence']
                order_number = row['Order Number']

                # Append the original row (no changes)
                amplified_rows.append(row)

                # If the row is amplified
                if random.random() < amplification_probability:
                    row_copy = row.copy()

                    # Apply polymerase errors with a probability
                    mutated_sequence = random_errors_polymerase(sequence, error_rate)
                    row_copy['Nucleotide Sequence'] = mutated_sequence

                    # Check if the sequence was actually mutated and assign a unique order number if it was
                    if mutated_sequence != sequence:
                        row_copy['Order Number'] = next_order_number
                        next_order_number += 1  # Ensure the next unique number is used

                    # Add the modified row (amplified and possibly mutated)
                    amplified_rows.append(row_copy)

            # Update the DataFrame for the next cycle
            self.df = pd.DataFrame(amplified_rows, columns=self.df.columns)
            print(f"Cycle {cycle + 1} complete. The DataFrame now has {len(self.df)} rows.")

        # Save the resulting amplified DataFrame to a CSV file
        self.df.to_csv(output_filename, index=False)
        print(f"File '{output_filename}' has been created with {len(self.df)} rows.")

    def true_umis_analyze(self):
        if self.df is not None:
            loci_counts = self.df['Genetic Loci'].value_counts()
            print("Counts of rows with the same Genetic Loci:")
            print(loci_counts)
        else:
            print("Error: True UMIs have not been created yet.")

    def pcr_analyze(self):
        if self.df is not None:
            unique_order_numbers = self.df['Order Number'].nunique()
            print(f"Number of unique Order Numbers: {unique_order_numbers}")
        else:
            print("Error: True UMIs have not been created yet.")
