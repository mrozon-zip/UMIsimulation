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

        # Assign molecule numbers from 1 to the number_of_rows to each sequence
        self.df['Molecule'] = range(1, self.number_of_rows + 1)

        # Reorder the columns to match the desired output
        self.df = self.df[['Molecule', 'Nucleotide Sequence', 'Genetic Loci']]

        # Output the resulting DataFrame to a CSV file
        self.df.to_csv(output_filename, index=False)
        print(f"File '{output_filename}' has been created with {self.number_of_rows} unique rows and sequence length of {self.length}.")

    def amplify_with_errors(self, amplification_probability, error_rate, error_types, amplification_cycles, output_filename='amplified_UMIs.csv'):
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
                molecule_number = row['Molecule']

                # Append the original row (no changes)
                amplified_rows.append(row)

                # If the row is amplified
                if random.random() < amplification_probability:
                    row_copy = row.copy()

                    # Apply polymerase errors with a probability
                    mutated_sequence = random_errors(sequence, error_rate, error_types)
                    row_copy['Nucleotide Sequence'] = mutated_sequence

                    # Check if the sequence was actually mutated and assign a unique molecule number if it was
                    if mutated_sequence != sequence:
                        row_copy['Molecule'] = next_molecule_number
                        next_molecule_number += 1  # Ensure the next unique number is used

                    # Add the modified row (amplified and possibly mutated)
                    amplified_rows.append(row_copy)

            # Update the DataFrame for the next cycle
            self.df = pd.DataFrame(amplified_rows, columns=self.df.columns)
            print(f"Cycle {cycle + 1} complete. The DataFrame now has {len(self.df)} rows.")

        # Save the resulting amplified DataFrame to a CSV file
        self.df.to_csv(output_filename, index=False)
        print(f"File '{output_filename}' has been created with {len(self.df)} rows.")

    def true_UMIs_analyze(self):
        if self.df is not None:
            loci_counts = self.df['Genetic Loci'].value_counts()
            print("Counts of rows with the same Genetic Loci:")
            print(loci_counts)
        else:
            print("Error: True UMIs have not been created yet.")

    def PCR_analyze(self):
        if self.df is not None:
            unique_molecule_numbers = self.df['Molecule'].nunique()
            print(f"Number of unique Molecule numbers: {unique_molecule_numbers}")
        else:
            print("Error: True UMIs have not been created yet.")
