from classes import Denoiser
from classes import simPCR

# Example usage
simulator = simPCR(length=12, number_of_rows=100)
simulator.create_true_UMIs()
error_types = {
    'substitution': 0.6,  # 60% chance of substitution
    'deletion': 0.2,      # 20% chance of deletion
    'insertion': 0.2      # 20% chance of insertion
}
simulator.amplify_with_errors(amplification_probability=1, error_rate=0, error_types=error_types, amplification_cycles=8)
simulator.PCR_analyze()

# Create an instance of the Denoiser class
denoiser = Denoiser(csv_file='amplified_UMIs.csv')

# File paths for the data
true_umis_file = "true_UMIs.csv"

method = 1
do_denoising = False

if do_denoising == True:
    print("I am denoising")
    # simple
    if method == 0:
        denoised_file = "simple_result.csv"
        # Set a threshold for filtering sequences
        threshold_value = 63  # Example threshold

        # Apply the simple denoising method with the specified threshold
        filtered_results = denoiser.simple(threshold_value)

        # Display the number of rows in the filtered DataFrame
        if filtered_results is not None:
            print(f"Number of rows in the collapsed DataFrame: {len(filtered_results)}")

        # Call the analysis method
        denoiser.analysis(denoised_file, true_umis_file)
    # directional
    elif method == 1:
        denoised_file = "directional_results.csv"
        # Step 1: Run the directional_networks method to create networks
        before_graph, after_graph, unique_molecules = denoiser.directional_networks(show=2)

        # Step 2: Use the networks_resolver method to analyze central nodes in the networks
        central_nodes_df = denoiser.networks_resolver(after_graph, toggle="central_node")

        # Step 3: Use the node_probe method on the after_graph
        denoiser.node_probe(after_graph, tier1=5, tier2=3)

        # Call the analysis method
        denoiser.analysis(denoised_file, true_umis_file)

elif do_denoising == False:
    print("I am waiting")
