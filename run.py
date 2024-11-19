import pandas as pd

from classes import Denoiser

# Create an instance of the Denoiser class
denoiser = Denoiser(csv_file='amplified_UMIs.csv')

method = 1

if method == 0:
    # Set a threshold for filtering sequences
    threshold_value = 250  # Example threshold

    # Apply the simple denoising method with the specified threshold
    filtered_results = denoiser.simple(threshold_value)

    # Display the number of rows in the filtered DataFrame
    if filtered_results is not None:
        print(f"Number of rows in the collapsed DataFrame: {len(filtered_results)}")
elif method == 1:
    # Step 1: Run the directional_networks method to create networks
    directional_denoiser = denoiser.directional_networks()

    graph = directional_denoiser[0]

    # Step 2: Use the networks_resolver method to analyze central nodes in the networks
    central_nodes_df = denoiser.networks_resolver(graph, toggle="central_node")

    denoiser.analysis("simple_denoiser.csv", "true_UMIs.csv")

    # Display the DataFrame with central nodes for verification
    print(central_nodes_df)

