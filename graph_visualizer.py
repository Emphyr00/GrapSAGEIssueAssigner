import networkx as nx
import matplotlib.pyplot as plt

# Load the graph from a GML file
gml_file_path = 'graph_small_company.gml'  # replace with your GML file path
G = nx.read_gml(gml_file_path)

# Draw the graph
pos = nx.spring_layout(G)  # You can use other layouts like circular_layout, shell_layout, etc.
plt.figure(figsize=(12, 8))

# Draw the nodes with smaller size
nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue')
nx.draw_networkx_edges(G, pos, width=2)

# Use the 'keywords' attribute for labels
labels = nx.get_node_attributes(G, 'keywords')  # Replace 'keywords' with the actual attribute name in your GML file
nx.draw_networkx_labels(G, pos, labels, font_size=10)

# Show the graph
plt.title("Graph Visualization from GML")
plt.show()

# Print out the content of the nodes
for node in G.nodes(data=True):
    keywords = node[1].get('keywords', 'No keywords')
    print(f"Node {node[0]}: Keywords = {keywords}")