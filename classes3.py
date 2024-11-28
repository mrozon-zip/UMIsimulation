import pandas as pd
import random

random.seed(47)

class simPCR:
    def __init__(self, length, number_of_rows):
        self.length = length
        self.number_of_rows = number_of_rows
        self.df = None  # DataFrame to store the UMIs
        self.old_df = None
        self.new_df = None

    def create_true_UMIs(self, output_filename='true_UMIs.csv'):
        nucleotides = ['A', 'C', 'G', 'T']
        sequences = set()

        while len(sequences) < self.number_of_rows:
            sequence = ''.join(random.choices(nucleotides, k=self.length))
            sequences.add(sequence)

        self.df = pd.DataFrame(list(sequences), columns=['Nucleotide Sequence'])
        self.df['edit distance'] = 0
        self.df['amount'] = 1

        self.df.to_csv(output_filename, index=False)
        print(f"File '{output_filename}' created with {self.number_of_rows} unique rows.")

    def amplify_old(self, amplification_probability, error_rate, error_types, amplification_cycles,
                    output_filename='amplified_UMIs_old.csv', output_filename2='amplified_UMIs_new.csv'):
        df_old = self.df.copy()
        df_new = self.df.copy()
        df_old_restricted = df_old.copy()

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
            for i, row in df_old_restricted.iterrows():
                sequence = row['Nucleotide Sequence']
                amount_new = df_new.loc[df_new['Nucleotide Sequence'] == sequence, 'amount'].values[0]

                if random.random() < amplification_probability:
                    mutated_sequence = random_errors(sequence)
                    if mutated_sequence == sequence:
                        # Increment amount for the existing sequence in df_new
                        if i in range(10):
                            print(amount_new)
                        amount_new += 1
                        df_new.loc[df_new['Nucleotide Sequence'] == sequence, 'amount'] = amount_new
                        # Add mutated sequence as a new row in df_old with amount = 1
                        df_old = pd.concat([df_old, pd.DataFrame({
                            'Nucleotide Sequence': [mutated_sequence],
                            'edit distance': [0],  # Adjust if needed
                            'amount': [1]
                        })], ignore_index=True)
                    else:
                        # Add mutated sequence as a new row in df_new with amount = 1
                        df_new = pd.concat([df_new, pd.DataFrame({
                            'Nucleotide Sequence': [mutated_sequence],
                            'edit distance': [0],  # Adjust if needed
                            'amount': [1]
                        })], ignore_index=True)

                        # Add mutated sequence as a new row in df_old with amount = 1
                        df_old = pd.concat([df_old, pd.DataFrame({
                            'Nucleotide Sequence': [mutated_sequence],
                            'edit distance': [0],  # Adjust if needed
                            'amount': [1]
                        })], ignore_index=True)

                # Restrict df_old_restricted to current df_old for the next cycle
            df_old_restricted = df_old.copy()
            print(f"Cycle {cycle + 1}: {len(df_old)} rows - OLD")
            print(f"Cycle {cycle + 1}: {len(df_new)} rows - NEW")
            print(df_new['Nucleotide Sequence'].nunique())

        # Collapse duplicates and count occurrences
        df_old = df_old.groupby('Nucleotide Sequence', as_index=False).agg({'amount': 'sum'})
        df_new = df_new.groupby('Nucleotide Sequence', as_index=False).agg({'amount': 'sum'})
        df_old.to_csv(output_filename, index=False)
        df_new.to_csv(output_filename2, index=False)
        print(f"File '{output_filename}' created with {len(df_old)} rows.")
        print(f"File '{output_filename2}' created with {len(df_new)} rows.")


    # def amplify_new(self, amplification_probability, error_rate, error_types, amplification_cycles,
    #                 output_filename='amplified_UMIs_new.csv'):
    #     def introduce_errors(sequence):
    #         nucleotides = ['A', 'C', 'G', 'T']
    #         sequence_list = list(sequence)
    #         i = 0
    #
    #         while i < len(sequence_list):
    #             if random.random() < error_rate:
    #                 error_type = random.choices(
    #                     ['substitution', 'deletion', 'insertion'],
    #                     weights=[error_types['substitution'], error_types['deletion'], error_types['insertion']],
    #                     k=1
    #                 )[0]
    #
    #                 if error_type == 'substitution':
    #                     sequence_list[i] = random.choice([n for n in nucleotides if n != sequence_list[i]])
    #                     i += 1
    #                 elif error_type == 'deletion':
    #                     del sequence_list[i]
    #                 elif error_type == 'insertion':
    #                     sequence_list.insert(i + 1, random.choice(nucleotides))
    #                     i += 2
    #             else:
    #                 i += 1
    #
    #         return ''.join(sequence_list)
    #
    #     for cycle in range(amplification_cycles):
    #         new_rows = []
    #
    #         for _, row in self.new_df.iterrows():
    #             initial_amount = int(row['amount'])
    #             for _ in range(initial_amount):  # Process each sequence based on its current amount
    #                 sequence = row['Nucleotide Sequence']
    #
    #                 # Amplify with a probability
    #                 if random.random() < amplification_probability:
    #                     mutated_sequence = introduce_errors(sequence)
    #
    #                     edit_distance = levenshtein(mutated_sequence, sequence)
    #
    #                     if mutated_sequence == sequence:
    #                         # If the sequence remains the same, increment its amount
    #                         self.new_df.loc[self.new_df['Nucleotide Sequence'] == sequence, 'amount'] += 1
    #                     else:
    #                         # Add new mutated sequence
    #                         new_rows.append({
    #                             'Nucleotide Sequence': mutated_sequence,
    #                             'amount': 1,
    #                             'edit distance': edit_distance
    #                         })
    #
    #         # Add all new rows to the main DataFrame
    #         for new_row in new_rows:
    #             existing_row = self.new_df[self.new_df['Nucleotide Sequence'] == new_row['Nucleotide Sequence']]
    #             if not existing_row.empty:
    #                 self.new_df.loc[self.new_df['Nucleotide Sequence'] == new_row['Nucleotide Sequence'], 'amount'] += \
    #                 new_row['amount']
    #             else:
    #                 self.new_df = pd.concat([self.new_df, pd.DataFrame([new_row])], ignore_index=True)
    #
    #         print(f"Cycle {cycle + 1}: {len(self.new_df)} rows after amplification.")
    #
    #     # Keep only the necessary columns and save the output
    #     self.new_df = self.new_df[['Nucleotide Sequence', 'amount', 'edit distance']]
    #     self.new_df.to_csv(output_filename, index=False)
    #     print(f"File '{output_filename}' created with {len(self.new_df)} rows.")