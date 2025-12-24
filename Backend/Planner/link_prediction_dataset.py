import networkx as nx
import torch


class link_prediction_dataset:
    def __init__(self, graph, node_features):
        self.graph = graph
        self.node_features = node_features
        self.node_to_idx = {}
        self.idx_to_node = {}

    def create_dataset(self):
        G_simple = nx.Graph(self.graph)
        node_list = list(G_simple)
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}
        X = torch.zeros((len(node_list), 2), dtype=torch.float)
        for idx, node in enumerate(node_list):
            feats = self.node_features[node]
            X[idx] = torch.tensor(
                [feats["degree"], feats["dist_to_main_road"] / 1000], dtype=torch.float
            )
        X = (X - X.mean(dim=0)) / (X.std(dim=0) + 1e-6)
        undirected_edges = [
            (node_to_idx[u], node_to_idx[v]) for u, v in G_simple.edges()
        ]
        self.node_to_idx = node_to_idx
        self.idx_to_node = {v: k for k, v in node_to_idx.items()}
