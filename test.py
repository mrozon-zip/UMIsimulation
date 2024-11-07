from classes import Denoiser

# Create an instance of the Denoiser class
denoiser = Denoiser(csv_file='amplified_UMIs.csv')

# Execute the directional method to create networks
networks, combined_network = denoiser.directional()

# Save the central nodes to a CSV file
denoiser.save_central_nodes(networks)

# Visualize the individual networks for each genetic loci
denoiser.visualize_individual_networks(networks)

# Visualize the combined network
if combined_network:
    denoiser.visualize_combined_network(combined_network)
