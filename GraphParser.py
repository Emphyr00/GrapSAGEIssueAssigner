import pandas as pd
import networkx as nx
from tqdm import tqdm
from dask import dataframe as dd
import ast

class GraphParser:
    csv_file = None
    df = None
    graph = nx.DiGraph()  # Directed graph
    valid_keywords = set()  # Set of valid keywords based on unique classes
    label_dict = {}  # Dictionary to map each unique repo to a unique number

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
    def create_label_dict():
        try:
            unique_repos = GraphParser.df['repo'].dropna().unique().compute()
            GraphParser.label_dict = {repo: idx for idx, repo in enumerate(unique_repos)}
            print(f"Label dictionary: {GraphParser.label_dict}")
        except Exception as e:
            print(f"An error occurred while creating label dictionary: {e}")

    @staticmethod
    def process_data(sample_percentage=100):
        try:
            GraphParser.df = GraphParser.df.compute()

            if sample_percentage < 100:
                GraphParser.df = GraphParser.df.sample(frac=sample_percentage / 100.0)

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
            classes = [kw for kw in keywords if kw in GraphParser.valid_keywords]

            # Ensure consistent label for the repo using label dictionary
            label = GraphParser.label_dict[repo]

            # Add node with all keywords as features and numeric label
            GraphParser.graph.add_node(index, repo_label=label, label=label, keywords=keywords)
            # print(f"Added node: {index} with label: {label} and keywords: {keywords}")

            for other_index, other_row in GraphParser.df.iterrows():
                if index >= other_index:
                    continue

                other_repo = other_row['repo']
                other_keywords = ast.literal_eval(other_row['keywords'])
                other_classes = [kw for kw in other_keywords if kw in GraphParser.valid_keywords]

                shared_classes = set(classes).intersection(set(other_classes))

                if repo == other_repo and len(shared_classes) >= 2:
                    GraphParser.graph.add_edge(index, other_index, shared_classes=list(shared_classes))
                    # print(f"Added edge between {index} and {other_index} with shared classes: {shared_classes} (same repo)")
                elif repo != other_repo and len(shared_classes) >= 3:
                    GraphParser.graph.add_edge(index, other_index, shared_classes=list(shared_classes))
                    # print(f"Added edge between {index} and {other_index} with shared classes: {shared_classes} (different repos)")

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
            # Ensure the 'label' attribute matches 'repo_label'
            for node, data in GraphParser.graph.nodes(data=True):
                data['label'] = data['repo_label']
            nx.write_gml(GraphParser.graph, output_file)
        except Exception as e:
            print(f"An error occurred while saving the graph: {e}")

    @staticmethod
    def main(csv_file, sample_percentage=100):
        GraphParser.csv_file = csv_file
        GraphParser.read_csv()
        GraphParser.extract_valid_keywords()
        GraphParser.create_label_dict()
        GraphParser.process_data(sample_percentage)
        GraphParser.save_graph('graph_even_company.gml')
        print(f"Number of nodes: {GraphParser.graph.number_of_nodes()}")
        print(f"Number of edges: {GraphParser.graph.number_of_edges()}")
        return GraphParser.graph

# Example usage:
# graph = GraphParser.main('path_to_your_file.csv', sample_percentage=10)
