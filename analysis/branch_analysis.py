import csv
import ast
import matplotlib.pyplot as plt

"""
A part of analysis module script. It generates a branching analysis presented in thesis.
"""

def load_csv_as_dicts(file_path):
    """
    Load a CSV file and return a list of dictionaries.
    The CSV file should include a column 'born' whose value is a string
    representation of a list of integers (e.g., "[1, 2, 3]").

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        List[dict]: A list of dictionaries with the 'born' values converted to lists.
    """
    data = []
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if 'born' in row:
                # Convert the string "[1, 2, 3]" into an actual list of integers.
                try:
                    row['born'] = ast.literal_eval(row['born'])
                except (ValueError, SyntaxError) as e:
                    # In case of conversion failure, leave the value as-is or handle appropriately.
                    print(f"Error parsing row: {row}. Error: {e}")
            data.append(row)
    return data


def plot_distribution(data):
    """
    Plot a stacked bar chart based on the 'born' lists in the data.
    The x-axis represents distinct cycle numbers extracted from all dictionaries.
    Each dictionary contributes its counts as one group in the stacked bar.

    For each dictionary in 'data', the 'born' key is expected to be a list of integers.

    Args:
        data (List[dict]): The list of dictionaries containing the 'born' data.

    The function displays a plot where:
      - X-axis: Cycle numbers (e.g., 1, 2, 3, …)
      - Y-axis: Count of occurrences per cycle.
      - Each bar is subdivided into contributions (groups) from each dictionary.
    """
    # Identify all distinct cycle numbers in the dataset.
    cycle_numbers = set()
    counts_per_group = []  # This will hold a dictionary for each group with cycle -> count.

    for group in data:
        counts = {}
        born_list = group.get('born', [])
        for num in born_list:
            counts[num] = counts.get(num, 0) + 1
            cycle_numbers.add(num)
        counts_per_group.append(counts)

    # Sort cycle numbers to use them as x-axis labels.
    cycle_numbers = sorted(cycle_numbers)

    # Prepare the bar plot.
    bar_width = 0.5
    bottoms = [0] * len(cycle_numbers)  # This stores the bottom heights for the stacked bars.

    # Plot each group's contribution as part of the stacked bar chart.
    for index, counts in enumerate(counts_per_group):
        # Retrieve counts for each cycle, defaulting to 0 if not present.
        values = [counts.get(cycle, 0) for cycle in cycle_numbers]
        plt.bar(cycle_numbers, values, bar_width, bottom=bottoms, label=f'Group {index + 1}')
        # Update bottoms for the next group's bar.
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    plt.xlabel('Cycle Number')
    plt.ylabel('Amount')
    plt.title('Distribution of Born Values by Cycle Number')
    plt.legend()
    plt.savefig('distribution.png')
    plt.show()


# Example usage:
if __name__ == "__main__":
    # Replace 'input.csv' with the path to your CSV file.
    file_path = '../archive/dump/results1/polonies_amplified.csv'
    data = load_csv_as_dicts(file_path)

    # Generate the distribution graph.
    plot_distribution(data)