from classes import Denoiser

# Example usage
# Load the DataFrame from a CSV file
denoiser = Denoiser(csv_file='amplified_UMIs.csv')  # Replace with your actual file path

# Apply the simple denoising method with a specified appearance threshold
appearance_threshold = 2  # Example threshold
filtered_results = denoiser.simple(appearance_threshold)

# Display the number of rows in the final DataFrame
if filtered_results is not None:
    print(f"Number of rows in the filtered DataFrame: {len(filtered_results)}")
