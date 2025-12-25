import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd


class NetworkExtractor:
    def __init__(self, place_name: str):
        self.place_name = place_name
        self.graph = None
        self.buildings = None

    def download_data(self):
        self.graph = ox.graph_from_place(self.place_name, network_type="all")
        print("downloaded graph")
        self.buildings = ox.features_from_place(
            self.place_name, tags={"building": True}
        )
        print("downloaded buildings")

        self.graph = ox.project_graph(self.graph)
        self.buildings = ox.projection.project_graph(self.buildings)

        print(f"total no of buildings {len(self.buildings)}")
        print(f"total no of nodes {len(self.graph.nodes)}")
        print(f"total no of edges {len(self.graph.edges)}")

    def visualize(self):
        pass

    def node_features(self):
        nodes, edges = ox.graph_to_gdfs(self.graph)
