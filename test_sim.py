from classes import simPCR
from classes import Denoiser

# Example usage
simulator = simPCR(length=24, number_of_rows=100)
simulator.create_true_UMIs()
simulator.true_UMIs_analyze()
error_types = {
    'substitution': 0.6,  # 60% chance of substitution
    'deletion': 0.2,      # 20% chance of deletion
    'insertion': 0.2      # 20% chance of insertion
}
simulator.amplify_with_errors(amplification_probability=0.9, error_rate=0.01, error_types=error_types, amplification_cycles=6)
simulator.PCR_analyze()

# Load the DataFrame from a CSV file
denoiser = Denoiser(csv_file='amplified_UMIs.csv')  # Replace with your actual file path

# Set a threshold for filtering sequences
threshold_value = 10  # Example threshold

# Apply the simple denoising method with the specified threshold
filtered_results = denoiser.simple(threshold_value)

# Display the number of rows in the filtered DataFrame
if filtered_results is not None:
    print(f"Number of rows in the collapsed DataFrame: {len(filtered_results)}")
