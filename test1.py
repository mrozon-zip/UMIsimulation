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
central_nodes_data = []
networks = {}
networks_count = 0
joint_graph = nx.Graph()        # Initialize the joint graph

for loci, group in result_df.groupby('Genetic Loci'):
    g = f"G_{loci}"
    graphs[g] = nx.Graph()  # Create a new graph and assign it to G_i

    for i, row_a in group.iterrows():
        graphs[g].add_node(f"{i}",
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
                        graphs[g].add_edge(f"{i}", f"{j}")  # Connect from node a to node b
    networks[loci] = graphs[g]

    nx.draw_spring(graphs[g], with_labels=True)
    plt.show()
    networks_amount = int(nx.number_connected_components(graphs[g]))
    networks_count += networks_amount
    print(f"Number of networks in group {loci} : {networks_amount}")
print(f"Total amount of networks : {networks_count}")

for loci, graph in networks.items():
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
            'Genetic loci': loci,
            'Central node ID': central_node_id,
            'Central node count': central_node_amount,
            'Network nodes count': total_amount
        })

# Iterate through each graph in the dictionary and add nodes and edges to the joint graph
for g_name, graph in graphs.items():
    # Add nodes and their attributes
    for node, data in graph.nodes(data=True):
        joint_graph.add_node(node, **data)  # Copy attributes as well

    # Add edges between nodes, maintaining attributes if any
    for u, v, edge_data in graph.edges(data=True):
        joint_graph.add_edge(u, v, **edge_data)  # Copy edge attributes if present

# Visualize the joint graph
nx.draw_spring(joint_graph, with_labels=True)
plt.show()
print(f"Networks in total: {nx.number_connected_components(joint_graph)}")

# Create a DataFrame from the collected data
central_nodes_df = pd.DataFrame(central_nodes_data)

# Save to CSV
central_nodes_df.to_csv('central_nodes.csv', index=False)
print(f"Central nodes saved to 'central_nodes.csv' ")

