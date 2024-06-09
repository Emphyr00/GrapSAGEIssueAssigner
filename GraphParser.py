import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from dask import dataframe as dd
from dask.diagnostics import ProgressBar
import ast

class GraphParser:
    csv_file = None
    df = None
    graph = nx.DiGraph()  # Directed graph
    valid_keywords = set()  # Set of valid keywords based on unique classes

    @staticmethod
    def read_csv():
        try:
            GraphParser.df = dd.read_csv(GraphParser.csv_file, assume_missing=True)
        except pd.errors.ParserError as e:
            print(f"ParserError: {e}")
        except Exception as e:
            print(f"An error occurred while reading the CSV file: {e}")

    @staticmethod
    def extract_valid_keywords():
        try:
            # Extract all unique classes
            unique_classes = GraphParser.df['class'].dropna().unique().compute()
            GraphParser.valid_keywords = set(unique_classes)
            print(f"Valid keywords based on classes: {GraphParser.valid_keywords}")
        except Exception as e:
            print(f"An error occurred while extracting valid keywords: {e}")

    @staticmethod
    def process_data():
        try:
            GraphParser.df = GraphParser.df.compute()
            num_rows = len(GraphParser.df)
        except Exception as e:
            print(f"An error occurred while computing the DataFrame: {e}")
            return

        # Initialize the progress bar
        pbar = tqdm(total=num_rows, desc="Processing Rows")

        def add_node_and_edges(row):
            index = row.name
            repo = row['repo']
            keywords = ast.literal_eval(row['keywords'])
            classes = row['class'].split(',') if row['class'] else []

            # Add node with all keywords as features and repo as label
            GraphParser.graph.add_node(index, keywords=keywords, label=repo)

            for other_index, other_row in GraphParser.df.iterrows():
                if index >= other_index:
                    continue

                other_repo = other_row['repo']
                other_keywords = ast.literal_eval(other_row['keywords'])
                other_classes = other_row['class'].split(',') if other_row['class'] else []

                # Filter keywords for edge creation based on valid_keywords
                filtered_keywords = [kw for kw in keywords if kw in GraphParser.valid_keywords]
                filtered_other_keywords = [kw for kw in other_keywords if kw in GraphParser.valid_keywords]

                shared_keywords = set(filtered_keywords).intersection(set(filtered_other_keywords))
                shared_classes = set(classes).intersection(set(other_classes))

                if repo == other_repo and len(shared_classes) >= 2:
                    GraphParser.graph.add_edge(index, other_index, keywords=list(shared_keywords))
                elif repo != other_repo and len(shared_classes) >= 3:
                    GraphParser.graph.add_edge(index, other_index, keywords=list(shared_keywords))

            pbar.update(1)

        try:
            # Apply the function in parallel
            GraphParser.df.apply(add_node_and_edges, axis=1)
        except Exception as e:
            print(f"An error occurred during processing: {e}")
        finally:
            # Close the progress bar
            pbar.close()

    @staticmethod
    def save_graph(output_file):
        try:
            nx.write_gml(GraphParser.graph, output_file)
        except Exception as e:
            print(f"An error occurred while saving the graph: {e}")

    @staticmethod
    def visualize_graph():
        try:
            pos = nx.spring_layout(GraphParser.graph, seed=42)  # for reproducible layout
            plt.figure(figsize=(15, 15))

            # Draw nodes
            nx.draw_networkx_nodes(GraphParser.graph, pos, node_size=500, node_color='skyblue')

            # Draw repository labels (node labels)
            repo_labels = nx.get_node_attributes(GraphParser.graph, 'label')
            nx.draw_networkx_labels(GraphParser.graph, pos, labels=repo_labels, font_size=10, verticalalignment='bottom')

            # Draw keywords above nodes
            keyword_labels = {i: ', '.join(data['keywords']) for i, data in GraphParser.graph.nodes(data=True)}
            nx.draw_networkx_labels(GraphParser.graph, pos, labels=keyword_labels, font_size=8, verticalalignment='top')

            # Draw edges
            nx.draw_networkx_edges(GraphParser.graph, pos, width=1.0, alpha=0.5)

            # Display the graph
            plt.title("Keyword Graph")
            plt.show()
        except Exception as e:
            print(f"An error occurred while visualizing the graph: {e}")

    @staticmethod
    def main(csv_file):
        GraphParser.csv_file = csv_file
        GraphParser.read_csv()
        GraphParser.extract_valid_keywords()
        GraphParser.process_data()
        GraphParser.save_graph('keyword_graph_final3.gml')
        print(f"Number of nodes: {GraphParser.graph.number_of_nodes()}")
        print(f"Number of edges: {GraphParser.graph.number_of_edges()}")
        GraphParser.visualize_graph()

# Usage example:
# GraphParser.main('path_to_your_file.csv')
