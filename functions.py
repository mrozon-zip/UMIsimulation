import pandas as pd
import random


# Function to create true UMIs with specified sequence length and number of rows
def create_true_UMIs(length, number_of_rows, output_filename='true_UMIs.csv'):
    # Step 1: Generate random nucleotide sequences
    nucleotides = ['A', 'C', 'G', 'T']

    sequences = []
    for i in range(number_of_rows):
        sequence = ''.join(random.choices(nucleotides, k=length))
        sequences.append([sequence])

    # Step 2: Assign random genetic loci between 1 and 8 to each sequence
    genetic_loci = [random.randint(1, 8) for _ in range(number_of_rows)]

    # Create a DataFrame with sequences and their corresponding genetic loci
    df = pd.DataFrame(sequences, columns=['Nucleotide Sequence'])
    df['Genetic Loci'] = genetic_loci

    # Step 3: Sort the DataFrame by 'Genetic Loci' in increasing order
    df_sorted = df.sort_values(by='Genetic Loci').reset_index(drop=True)

    # Step 4: Assign order numbers from 1 to the number_of_rows to each sequence
    df_sorted['Order Number'] = range(1, number_of_rows + 1)

    # Step 5: Reorder the columns to match the desired output
    df_final = df_sorted[['Order Number', 'Nucleotide Sequence', 'Genetic Loci']]

    # Output the resulting DataFrame to a CSV file
    df_final.to_csv(output_filename, index=False)

    print(f"File '{output_filename}' has been created with {number_of_rows} rows and sequence length of {length}.")


def amplify_with_errors(df, amplification_probability, error_rate, amplification_cycles,
                        output_filename='amplified_UMIs.csv'):
    """
    Function to amplify rows based on probability, introduce errors, and assign unique order numbers to sequences with errors.

    Parameters:
    - df: DataFrame containing the initial sequences and order numbers
    - amplification_probability: Probability of amplification for each sequence in each cycle
    - error_rate: Probability of introducing an error during amplification
    - amplification_cycles: Number of amplification cycles to perform
    - output_filename: Name of the output CSV file to save the results

    Output:
    - Saves the amplified DataFrame to a CSV file
    """

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
        next_order_number = get_next_order_number(df)

        # Iterate through each row and check for amplification and errors
        for index, row in df.iterrows():
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
        df = pd.DataFrame(amplified_rows, columns=df.columns)
        print(f"Cycle {cycle + 1} complete. The DataFrame now has {len(df)} rows.")

    # Save the resulting amplified DataFrame to a CSV file
    df.to_csv(output_filename, index=False)

    print(f"File '{output_filename}' has been created with {len(df)} rows.")
