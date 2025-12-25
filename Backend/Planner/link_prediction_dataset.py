import networkx as nx
import torch
from torch_geometric.data import Data
import numpy as np
from torch_geometric.utils import negative_sampling


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
        edge_index = torch.tensor(
            undirected_edges + [(v, u) for u, v in undirected_edges], dtype=torch.long
        )
        data = Data(x=X, edge_index=edge_index)
        data.num_nodes = len(node_list)
        return data

    def split_edges(
        self,
        data: Data,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        E = data.edge_index.cpu().numpy()
        pairs = set()
        for u, v in zip(E[0], E[1]):
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            pairs.add(a, b)
        undirected_edges = np.array(list(pairs), dtype=np.int64)
        num_edges = undirected_edges.shape[0]
        perm = rng.permutation(num_edges)
        num_val = int(num_edges * val_ratio)
        num_test = int(num_edges * test_ratio)
        num_train = num_edges - num_val - num_test

        train_udir = undirected_edges[perm[:num_train]]
        val_udir = undirected_edges[perm[num_train : num_train + num_val]]
        test_udir = undirected_edges[perm[num_train + num_val :]]

        def to_bidir(edge_pairs: np.ndarray) -> torch.Tensor:
            if edge_pairs.size == 0:
                return torch.empty((2, 0), dtype=torch.long)
            u = edge_pairs[:, 0]
            v = edge_pairs[:, 1]
            bidir = np.vstack([np.concatenate([u, v]), np.concatenate([v, u])])
            return torch.tensor(bidir, dtype=torch.long)

        train_pos = to_bidir(train_udir)
        val_pos = to_bidir(val_udir)
        test_pos = to_bidir(test_udir)

        train_neg = negative_sampling(
            edge_index=train_pos,
            num_nodes=data.num_nodes,
            num_neg_samples=train_pos.size(1),
        )
        val_neg = negative_sampling(
            edge_index=torch.cat([train_pos, val_pos], dim=1),
            num_nodes=data.num_nodes,
            num_neg_samples=val_pos.size(1),
        )
        test_neg = negative_sampling(
            edge_index=torch.cat([train_pos, val_pos, test_pos], dim=1),
            num_nodes=data.num_nodes,
            num_neg_samples=test_pos.size(1),
        )

        return {
            "train_pos": train_pos,
            "train_neg": train_neg,
            "val_pos": val_pos,
            "val_neg": val_neg,
            "test_pos": test_pos,
            "test_neg": test_neg,
        }
