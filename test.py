import pandas as pd
import numpy as np

# Example DataFrame
data = {
    "sentence": ["hello world", "data science", "python programming"],
    "amount": [2, 3, 1]
}
df = pd.DataFrame(data)

# Set probabilities for the operations
row_probability = 0.9  # Probability to decide whether to process the row
letter_probability = 0.03  # Probability to decide whether to capitalize a letter
k_cycles = 3  # Number of cycles


def process_dataframe(df, row_probability, letter_probability, k_cycles):
    for cycle in range(k_cycles):
        # Make a copy of the original DataFrame to iterate over
        original_df = df.copy()

        # List to store new rows
        new_rows = []

        for index, row in original_df.iterrows():
            if np.random.rand() > row_probability:
                # Skip this row based on probability
                continue

            initial_amount = row['amount']
            new_amount = row['amount']  # Track if "amount" needs to be incremented

            for _ in range(initial_amount):
                sentence = row['sentence']
                new_sentence = list(sentence)
                changes_made = False

                for i, char in enumerate(new_sentence):
                    if char.islower() and np.random.rand() < letter_probability:
                        new_sentence[i] = char.upper()
                        changes_made = True

                if changes_made:
                    # Add a new row with the modified sentence
                    new_rows.append({
                        "sentence": ''.join(new_sentence),
                        "amount": initial_amount
                    })
                else:
                    # Increment the "amount" if no changes were made
                    new_amount += 1

            # Update the "amount" in the original row if necessary
            df.at[index, 'amount'] = new_amount

        # Add new rows to the DataFrame
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    return df


# Run the script
result_df = process_dataframe(df, row_probability, letter_probability, k_cycles)

print(result_df)


