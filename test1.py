import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from polyleven import levenshtein

# Create an instance of the Denoiser class
df = pd.read_csv('amplified_UMIs.csv')

# Count occurrences of each Molecule
count_df = df['Molecule'].value_counts().reset_index()
count_df.columns = ['Molecule', 'amount']

# Merge the count back into the original DataFrame
merged_df = pd.merge(df, count_df, on='Molecule')

# Keep only the first occurrence of each Molecule
result_df = merged_df.drop_duplicates(subset=['Molecule'])
print(result_df)

# Dictionary to store graphs with dynamic names
graphs = {}
G = nx.Graph()    # joint graph
central_nodes_data = []
networks = {}

for loci, group in result_df.groupby('Genetic Loci'):
    graphs[f"G_{loci}"] = nx.Graph()  # Create a new graph and assign it to G_i

    for i, row_a in group.iterrows():
        graphs[f"G_{loci}"].add_node(f"{i}",
                                     sequence=row_a['Nucleotide Sequence'],
                                     amount=row_a['amount'],
                                     genetic_loci=row_a['Genetic Loci'],
                                     molecule=row_a['Molecule'])
        for j, row_b in group.iterrows():
            if i != j:  # Prevent self-connections
                value_a = row_a['amount']
                value_b = row_b['amount']

                # Condition: Check value condition
                if value_a >= 2 * value_b - 1:
                    # Check edit distance condition
                    edit_distance = levenshtein(row_a['Nucleotide Sequence'], row_b['Nucleotide Sequence'])
                    if edit_distance == 1:
                        graphs[f"G_{loci}"].add_edge(f"{i}", f"{j}")  # Connect from node a to node b
                        G.add_edge(f"{i}", f"{j}")
    networks[loci] = graphs[f"G_{loci}"]

    nx.draw_spring(graphs[f"G_{loci}"], with_labels=True)
    plt.show()
    print(nx.number_connected_components(graphs[f"G_{loci}"]))

## ANOTHER METHOD HERE
    # input: joint graph G
    # output: csv file with info about all nodes
for loci, graph in networks.items():
    # Identify strongly connected components
    for component in nx.connected_components(graph):
        subgraph = graph.subgraph(component)
        # Identify the central node (the node with the maximum 'amount' in this subgraph)
        central_node = max(subgraph.nodes(data=True), key=lambda x: x[1]['amount'])
        central_node_id = central_node[0]
        central_node_molecule = central_node[1]['molecule']
        central_node_amount = central_node[1]['amount']
        sequence = central_node[1]['sequence']
        loci = central_node[1]['genetic_loci']

        # Sum amounts of all nodes connected to the central node
        total_amount = central_node_amount  # Start with the central node's amount
        connected_nodes = nx.descendants(subgraph, central_node_id)  # Get all connected nodes

        # Add the amount of each connected node
        for node in connected_nodes:
            total_amount += subgraph.nodes[node]['amount']

        # Append the result
        central_nodes_data.append({
            'Sequence': sequence,
            'loci': loci,
            'Central Node': central_node_id,
            'Central Amount': central_node_amount,
            'Total Amount': total_amount
        })

# Create a DataFrame from the collected data
central_nodes_df = pd.DataFrame(central_nodes_data)
print(central_nodes_df)

# Save to CSV
central_nodes_df.to_csv('central_nodes.csv', index=False)
print(f"Central nodes saved to 'central_nodes.csv' ")

nx.draw_spring(G, with_labels=True)
plt.show()
print(nx.number_connected_components(G))

