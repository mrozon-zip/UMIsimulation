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

        # Initialize the 'root', 'Type', and 'Edit Distance' columns
        self.df['root'] = self.df['Molecule'].astype(str)  # Copy Molecule to root
        self.df['Type'] = 'original'  # All initial sequences are original
        self.df['Edit Distance'] = 0  # Initialize the Edit Distance column

        # Reorder the columns to match the desired output
        self.df = self.df[['root', 'Molecule', 'Nucleotide Sequence', 'Genetic Loci', 'Type', 'Edit Distance']]

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
                molecule_number = row['Molecule']
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


import pandas as pd

import pandas as pd


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
            'Genetic Loci': 'first',  # Choose the first occurrence
            'Type': 'first'  # Choose the first occurrence
        }).reset_index()

        # Output the collapsed DataFrame to 'simple_denoiser_result_enhanced.csv'
        collapsed_df.to_csv('simple_denoiser_result_enhanced.csv', index=False)
        print(f"Enhanced results saved to 'simple_denoiser_result_enhanced.csv' with {len(collapsed_df)} rows.")

        # Create another DataFrame for the simple result with a new 'Molecule' column
        simple_result_df = collapsed_df.copy()
        simple_result_df['Molecule'] = range(1, len(simple_result_df) + 1)  # Generate new molecule numbers

        # Output the simple result to 'simple_denoiser_result.csv'
        simple_result_df[['Molecule', 'Nucleotide Sequence', 'Genetic Loci']].to_csv('simple_denoiser_result.csv',
                                                                                     index=False)
        print(f"Simple results saved to 'simple_denoiser_result.csv' with {len(simple_result_df)} rows.")

        return collapsed_df.reset_index(drop=True)  # Return the collapsed DataFrame
