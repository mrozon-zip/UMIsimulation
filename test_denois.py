from classes import Denoiser

# Load the DataFrame from a CSV file
denoiser = Denoiser(csv_file='amplified_UMIs.csv')  # Replace with your actual file path

# Set a threshold for filtering sequences
threshold_value = 10  # Example threshold

# Apply the simple denoising method with the specified threshold
filtered_results = denoiser.simple(threshold_value)

# Display the number of rows in the filtered DataFrame
if filtered_results is not None:
    print(f"Number of rows in the collapsed DataFrame: {len(filtered_results)}")