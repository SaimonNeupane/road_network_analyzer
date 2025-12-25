"""
Train models on Houten (downloaded dynamically via OSMnx) and save checkpoints + embeddings.

Outputs (in ./artifacts):
- drive_linkpred_model.pt            (new driveway link prediction)
- drive_linkpred_meta.pt
- drive_linkpred_houten_embeddings.pt

- cycleway_edgeclf_model.pt          (cycleway-on-existing-roads edge classifier)
- cycleway_edgeclf_meta.pt
- cycleway_edgeclf_houten_embeddings.pt

Designed for: RTX 4050 + 16GB RAM, long runtime allowed (<= 8h).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import networkx as nx
import osmnx as ox
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import negative_sampling


# ----------------------------- Config ---------------------------------


ox.settings.timeout = 180        # seconds (default is low)
ox.settings.use_cache = True
ox.settings.log_console = True

@dataclass
class TrainConfig:
    place_train: str = "Houten, Netherlands"
    artifacts_dir: str = "artifacts"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Link prediction (new roads)
    lp_epochs: int = 40
    lp_hidden: int = 64
    lp_lr: float = 1e-3

    # Cycleway edge classification (upgrade existing edges)
    cw_epochs: int = 15
    cw_hidden: int = 64
    cw_lr: float = 1e-3

    # Centrality approx (keep small; Houten is small anyway)
    edge_betw_k: int = 400

    # Repro
    seed: int = 42


# ----------------------------- Utils ----------------------------------


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def highway_to_str(val) -> str:
    """OSMnx can store highway as list; normalize to string token."""
    if val is None:
        return "unknown"
    if isinstance(val, (list, tuple)) and len(val) > 0:
        return str(val[0])
    return str(val)


def junction_is_roundabout(val) -> int:
    return int(str(val) == "roundabout")


def compute_connectivity_metrics(G: nx.MultiDiGraph) -> Dict[str, float]:
    Gs = nx.Graph(G)
    n = Gs.number_of_nodes()
    m = Gs.number_of_edges()
    if n == 0:
        return {"connectivity_score": 0.0, "largest_component_ratio": 0.0, "num_components": 0.0}

    comps = list(nx.connected_components(Gs))
    lcc = max(comps, key=len)
    lcc_ratio = len(lcc) / n
    connectivity_score = (m / n) * lcc_ratio
    return {
        "connectivity_score": float(connectivity_score),
        "largest_component_ratio": float(lcc_ratio),
        "num_components": float(len(comps)),
    }


def approximate_edge_betweenness(G: nx.MultiDiGraph, k: int = 400, weight: str = "length") -> Dict[Tuple[int, int], float]:
    """
    Returns betweenness keyed by (min(u,v), max(u,v)) for simple undirected edges.
    """
    Gs = nx.Graph(G)
    if Gs.number_of_edges() == 0:
        return {}
    eb = nx.edge_betweenness_centrality(Gs, k=min(k, Gs.number_of_nodes()), weight=weight, seed=42)
    return {(min(u, v), max(u, v)): float(val) for (u, v), val in eb.items()}


def build_drive_kdtree(drive_G: nx.MultiDiGraph):
    xy = np.array([(d["x"], d["y"]) for _, d in drive_G.nodes(data=True)], dtype=float)
    return (cKDTree(xy) if len(xy) else None), xy


# ----------------------------- OSM Download ---------------------------


class HoutenDownloader:
    def __init__(self, place: str):
        self.place = place
        self.G_drive: Optional[nx.MultiDiGraph] = None
        self.G_bike: Optional[nx.MultiDiGraph] = None
        self.buildings = None

    def download(self):
        print(f"Downloading OSM data for {self.place} ...")
        self.G_drive = ox.graph_from_place(self.place, network_type="drive", simplify=True)
        self.G_bike = ox.graph_from_place(self.place, network_type="bike", simplify=True)
        self.buildings = ox.features_from_place(self.place, tags={"building": True})

        # Project all to same CRS (UTM)
        self.G_drive = ox.project_graph(self.G_drive)
        self.G_bike = ox.project_graph(self.G_bike)
        self.buildings = ox.projection.project_gdf(self.buildings)

        print("✓ Download complete")
        print(f"  Drive: {len(self.G_drive.nodes)} nodes, {len(self.G_drive.edges)} edges")
        print(f"  Bike : {len(self.G_bike.nodes)} nodes, {len(self.G_bike.edges)} edges")
        print(f"  Buildings: {len(self.buildings)}")
        return self


# ----------------------------- Feature Builders ------------------------


class FeatureBuilder:
    """
    Build features that also exist in Place:
    - node: degree(street_count proxy), street_count if present, dist_to_drive, (optional) building_count proxy omitted to keep light
    - edge: highway token index, is_roundabout, length, betweenness (approx), plus global connectivity scalars

    We deliberately do NOT require maxspeed/lanes/cycleway tags as input features (Place may miss them).
    Cycleway tags are used only as TRAINING LABELS for the edge-classifier.
    """

    def __init__(self, G_drive: nx.MultiDiGraph):
        self.G_drive = G_drive
        self.drive_tree, _ = build_drive_kdtree(G_drive)
        self.conn = compute_connectivity_metrics(G_drive)
        self.edge_betw = approximate_edge_betweenness(G_drive, k=400)

        # highway vocabulary will be fitted on training edges
        self.highway_vocab: Dict[str, int] = {"unknown": 0}

    def fit_highway_vocab(self, G: nx.MultiDiGraph, top_k: int = 32):
        edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
        hw = edges["highway"].apply(highway_to_str)
        top = hw.value_counts().head(top_k).index.tolist()
        self.highway_vocab = {"unknown": 0}
        for i, token in enumerate(top, start=1):
            self.highway_vocab[token] = i
        return self.highway_vocab

    def node_features(self, G: nx.MultiDiGraph) -> Dict[int, Dict[str, float]]:
        feats = {}
        for node, data in G.nodes(data=True):
            x, y = float(data["x"]), float(data["y"])
            deg = float(nx.Graph(G).degree(node))  # undirected degree (street_count proxy)
            sc = float(data.get("street_count", deg))
            if self.drive_tree is None:
                dist_to_drive = 1000.0
            else:
                dist_to_drive, _ = self.drive_tree.query([x, y], k=1)
                dist_to_drive = float(dist_to_drive)

            feats[node] = {
                "degree": deg,
                "street_count": sc,
                "dist_to_drive": dist_to_drive,
                # global connectivity signals (same for all nodes, but harmless)
                "connectivity_score": self.conn["connectivity_score"],
                "largest_component_ratio": self.conn["largest_component_ratio"],
                "num_components": self.conn["num_components"],
            }
        return feats

    def edge_features_and_labels_cycleway(self, G_drive: nx.MultiDiGraph):
        """
        Returns:
          edge_keys: list of (u,v,key)
          edge_attr: torch.FloatTensor [E, F]
          edge_y: torch.FloatTensor [E] cycleway label (1/0)
        """
        edges_gdf = ox.graph_to_gdfs(G_drive, nodes=False, edges=True)

        edge_keys: List[Tuple[int, int, int]] = list(edges_gdf.index)
        E = len(edge_keys)

        # Build features
        # F = [length_km, is_roundabout, highway_idx_norm, betweenness]
        edge_attr = torch.zeros((E, 4), dtype=torch.float)

        # Labels: edge has cycleway tag or explicitly cycle-related
        y = torch.zeros((E,), dtype=torch.float)

        for i, (u, v, k) in enumerate(edge_keys):
            row = edges_gdf.loc[(u, v, k)]

            length = float(row.get("length", 0.0))
            junction = row.get("junction", None)
            highway = highway_to_str(row.get("highway", None))

            is_roundabout = float(junction_is_roundabout(junction))
            hw_idx = float(self.highway_vocab.get(highway, 0))
            hw_idx_norm = hw_idx / max(1.0, float(max(self.highway_vocab.values())))

            a, b = (u, v) if u < v else (v, u)
            betw = float(self.edge_betw.get((a, b), 0.0))

            edge_attr[i] = torch.tensor([length / 1000.0, is_roundabout, hw_idx_norm, betw], dtype=torch.float)

            # Label rule (Houten has these; Place may not, but labels only needed for training)
            has_cycleway = False
            for col in ["cycleway", "cycleway:left", "cycleway:right"]:
                if col in row and pd.notna(row[col]) and str(row[col]).strip() not in ["", "nan", "None"]:
                    has_cycleway = True
            # some datasets encode dedicated cycle infrastructure as highway=cycleway
            if highway == "cycleway":
                has_cycleway = True
            y[i] = 1.0 if has_cycleway else 0.0

        return edge_keys, edge_attr, y


# ----------------------------- PyG Dataset ----------------------------


class GraphDataset:
    def __init__(self, G: nx.MultiDiGraph, node_features: Dict[int, Dict[str, float]]):
        self.G = nx.Graph(G)  # simple undirected for GNN
        self.node_features = node_features
        self.node_to_idx: Dict[int, int] = {}
        self.idx_to_node: Dict[int, int] = {}

    def build(self) -> Data:
        nodes = list(self.G.nodes())
        self.node_to_idx = {n: i for i, n in enumerate(nodes)}
        self.idx_to_node = {i: n for n, i in self.node_to_idx.items()}

        # Node feature order
        cols = ["degree", "street_count", "dist_to_drive", "connectivity_score", "largest_component_ratio", "num_components"]
        X = torch.zeros((len(nodes), len(cols)), dtype=torch.float)
        for i, n in enumerate(nodes):
            f = self.node_features[n]
            X[i] = torch.tensor([float(f[c]) for c in cols], dtype=torch.float)

        # normalize
        X = (X - X.mean(dim=0)) / (X.std(dim=0) + 1e-6)

        edges = [(self.node_to_idx[u], self.node_to_idx[v]) for u, v in self.G.edges()]
        edge_index = torch.tensor(edges + [(v, u) for (u, v) in edges], dtype=torch.long).t().contiguous()

        data = Data(x=X, edge_index=edge_index)
        data.num_nodes = len(nodes)
        return data

    def split_edges(self, data: Data, val_ratio=0.1, test_ratio=0.2, seed=42):
        rng = np.random.default_rng(seed)
        E = data.edge_index.cpu().numpy()
        pairs = set()
        for u, v in zip(E[0], E[1]):
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            pairs.add((a, b))

        undirected_edges = np.array(list(pairs), dtype=np.int64)
        num_edges = undirected_edges.shape[0]
        perm = rng.permutation(num_edges)

        num_val = int(num_edges * val_ratio)
        num_test = int(num_edges * test_ratio)
        num_train = num_edges - num_val - num_test

        train_udir = undirected_edges[perm[:num_train]]
        val_udir = undirected_edges[perm[num_train:num_train + num_val]]
        test_udir = undirected_edges[perm[num_train + num_val:]]

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

        train_neg = negative_sampling(train_pos, num_nodes=data.num_nodes, num_neg_samples=train_pos.size(1))
        val_neg = negative_sampling(torch.cat([train_pos, val_pos], dim=1), num_nodes=data.num_nodes, num_neg_samples=val_pos.size(1))
        test_neg = negative_sampling(torch.cat([train_pos, val_pos, test_pos], dim=1), num_nodes=data.num_nodes, num_neg_samples=test_pos.size(1))

        return {"train_pos": train_pos, "train_neg": train_neg, "val_pos": val_pos, "val_neg": val_neg, "test_pos": test_pos, "test_neg": test_neg}


# ----------------------------- Models ---------------------------------


class EncoderSAGE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden: int = 64):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, 0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class LinkPredModel(torch.nn.Module):
    """Link prediction for new driveways (non-edges)."""

    def __init__(self, in_channels: int, hidden: int = 64):
        super().__init__()
        self.encoder = EncoderSAGE(in_channels, hidden)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden * 2, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid(),
        )

    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)

    def decode(self, z, edge_label_index):
        row, col = edge_label_index
        h = torch.cat([z[row], z[col]], dim=-1)
        return self.mlp(h).squeeze(-1)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)


class CyclewayEdgeClassifier(torch.nn.Module):
    """
    Edge classifier for "cycleway upgrade" on existing drive edges.
    Uses node embeddings + edge attributes.
    """

    def __init__(self, in_channels: int, edge_attr_dim: int = 4, hidden: int = 64):
        super().__init__()
        self.encoder = EncoderSAGE(in_channels, hidden)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden * 2 + edge_attr_dim, 96),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(96, 1),
            torch.nn.Sigmoid(),
        )

    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)

    def forward(self, x, edge_index, edge_u, edge_v, edge_attr):
        z = self.encode(x, edge_index)
        h = torch.cat([z[edge_u], z[edge_v], edge_attr], dim=-1)
        return self.mlp(h).squeeze(-1)


# ----------------------------- Training --------------------------------


@torch.no_grad()
def eval_linkpred(model: LinkPredModel, data: Data, splits, device: str):
    model.eval()
    pos = splits["val_pos"].to(device)
    neg = splits["val_neg"].to(device)
    pos_pred = model(data.x, data.edge_index, pos).detach().cpu().numpy()
    neg_pred = model(data.x, data.edge_index, neg).detach().cpu().numpy()
    preds = np.concatenate([pos_pred, neg_pred])
    labels = np.concatenate([np.ones_like(pos_pred), np.zeros_like(neg_pred)])
    return {
        "auc": float(roc_auc_score(labels, preds)),
        "ap": float(average_precision_score(labels, preds)),
    }


def train_linkpred(cfg: TrainConfig, data: Data, splits) -> LinkPredModel:
    device = cfg.device
    model = LinkPredModel(in_channels=data.num_node_features, hidden=cfg.lp_hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lp_lr)

    data = data.to(device)
    splits = {k: v.to(device) for k, v in splits.items()}

    best_auc = -1.0
    best_state = None

    for epoch in range(cfg.lp_epochs):
        model.train()
        opt.zero_grad()

        pos = splits["train_pos"]
        neg = splits["train_neg"]
        pos_pred = model(data.x, data.edge_index, pos)
        neg_pred = model(data.x, data.edge_index, neg)

        loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred)) + F.binary_cross_entropy(
            neg_pred, torch.zeros_like(neg_pred)
        )
        loss.backward()
        opt.step()

        if (epoch + 1) % 5 == 0:
            metrics = eval_linkpred(model, data, splits, device)
            print(f"[LP] epoch {epoch+1}/{cfg.lp_epochs} loss={loss.item():.4f} val_auc={metrics['auc']:.4f} val_ap={metrics['ap']:.4f}")
            if metrics["auc"] > best_auc:
                best_auc = metrics["auc"]
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"✓ LinkPred best val_auc={best_auc:.4f}")
    return model


def train_cycleway_edgeclf(cfg: TrainConfig, data: Data, edge_uvk, edge_attr: torch.Tensor, y: torch.Tensor) -> CyclewayEdgeClassifier:
    """
    Train on Houten drive edges. Balances positives/negatives.
    """
    device = cfg.device
    model = CyclewayEdgeClassifier(in_channels=data.num_node_features, edge_attr_dim=edge_attr.size(1), hidden=cfg.cw_hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.cw_lr)

    data = data.to(device)
    edge_attr = edge_attr.to(device)
    y = y.to(device)

    # map (u,v,key) -> node indices in pyg
    # edge_uvk are original OSM node ids; we need pyg indices
    # We'll use data._node_to_idx injected by caller via closure below.

    # Split indices
    n = y.numel()
    idx = torch.randperm(n, device=device)
    split = int(n * 0.8)
    train_idx = idx[:split]
    val_idx = idx[split:]

    # balance (downsample negatives)
    pos_idx = train_idx[y[train_idx] > 0.5]
    neg_idx = train_idx[y[train_idx] <= 0.5]
    if len(pos_idx) > 0:
        neg_idx = neg_idx[torch.randperm(len(neg_idx), device=device)[: min(len(neg_idx), len(pos_idx) * 2)]]
        train_idx = torch.cat([pos_idx, neg_idx])
        train_idx = train_idx[torch.randperm(len(train_idx), device=device)]

    def batch_iter(indices, bs=4096):
        for i in range(0, len(indices), bs):
            yield indices[i : i + bs]

    best_auc = -1.0
    best_state = None

    for epoch in range(cfg.cw_epochs):
        model.train()
        total_loss = 0.0
        batches = 0

        for b in batch_iter(train_idx, bs=4096):
            opt.zero_grad()
            eu = data.edge_u[b]
            ev = data.edge_v[b]
            pred = model(data.x, data.edge_index, eu, ev, edge_attr[b])
            loss = F.binary_cross_entropy(pred, y[b])
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            batches += 1

        if (epoch + 1) % 3 == 0:
            model.eval()
            with torch.no_grad():
                pred = model(data.x, data.edge_index, data.edge_u[val_idx], data.edge_v[val_idx], edge_attr[val_idx]).detach().cpu().numpy()
                lab = y[val_idx].detach().cpu().numpy()
                auc = float(roc_auc_score(lab, pred)) if len(np.unique(lab)) > 1 else 0.0
            print(f"[CW] epoch {epoch+1}/{cfg.cw_epochs} loss={total_loss/max(1,batches):.4f} val_auc={auc:.4f}")
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"✓ CyclewayEdgeClf best val_auc={best_auc:.4f}")
    return model


# ----------------------------- Saving ----------------------------------


def save_artifacts(prefix: str, model: torch.nn.Module, data: Data, idx_to_node: dict, meta: dict, out_dir: str):
    ensure_dir(out_dir)
    model_path = Path(out_dir) / f"{prefix}_model.pt"
    emb_path = Path(out_dir) / f"{prefix}_houten_embeddings.pt"
    meta_path = Path(out_dir) / f"{prefix}_meta.pt"

    torch.save(model.state_dict(), model_path)

    model.eval()
    with torch.no_grad():
        if hasattr(model, "encode"):
            z = model.encode(data.x, data.edge_index).detach().cpu()
        else:
            z = model.encoder(data.x, data.edge_index).detach().cpu()

    torch.save({"z": z, "idx_to_node": idx_to_node}, emb_path)
    torch.save(meta, meta_path)

    print(f"Saved: {model_path}")
    print(f"Saved: {emb_path}")
    print(f"Saved: {meta_path}")


# ----------------------------- Main ------------------------------------


def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)
    print(f"Device: {cfg.device}")

    dl = HoutenDownloader(cfg.place_train).download()

    # Feature builder uses DRIVE network as "main roads reference"
    fb = FeatureBuilder(dl.G_drive)
    highway_vocab = fb.fit_highway_vocab(dl.G_drive, top_k=32)

    # Build node features on DRIVE graph (for both tasks we operate on drive)
    node_feats = fb.node_features(dl.G_drive)
    ds = GraphDataset(dl.G_drive, node_feats)
    data = ds.build()
    splits = ds.split_edges(data, seed=cfg.seed)

    # ---------------- Link Prediction model (new driveways) -------------
    print("\n=== Training Drive Link Prediction (new roads) ===")
    lp_model = train_linkpred(cfg, data, splits)

    save_artifacts(
        prefix="drive_linkpred",
        model=lp_model,
        data=data.to(cfg.device),
        idx_to_node=ds.idx_to_node,
        meta={
            "task": "drive_link_prediction",
            "place_train": cfg.place_train,
            "highway_vocab": highway_vocab,
            "node_feature_cols": ["degree", "street_count", "dist_to_drive", "connectivity_score", "largest_component_ratio", "num_components"],
            "hidden_dim": cfg.lp_hidden,
            "seed": cfg.seed,
        },
        out_dir=cfg.artifacts_dir,
    )

    # ---------------- Cycleway edge classifier (upgrade edges) ----------
    print("\n=== Training Cycleway Edge Classifier (existing edges) ===")
    import pandas as pd  # needed for label rule
    edge_keys, edge_attr, y = fb.edge_features_and_labels_cycleway(dl.G_drive)

    # Build edge_u/edge_v as PyG indices aligned with edge_keys order
    edge_u = torch.tensor([ds.node_to_idx[u] for (u, v, k) in edge_keys], dtype=torch.long)
    edge_v = torch.tensor([ds.node_to_idx[v] for (u, v, k) in edge_keys], dtype=torch.long)

    # attach to data for trainer convenience
    data.edge_u = edge_u.to(cfg.device)
    data.edge_v = edge_v.to(cfg.device)

    cw_model = train_cycleway_edgeclf(cfg, data, edge_keys, edge_attr, y)

    save_artifacts(
        prefix="cycleway_edgeclf",
        model=cw_model,
        data=data.to(cfg.device),
        idx_to_node=ds.idx_to_node,
        meta={
            "task": "cycleway_edge_classification",
            "place_train": cfg.place_train,
            "highway_vocab": highway_vocab,
            "edge_attr_cols": ["length_km", "is_roundabout", "highway_idx_norm", "edge_betweenness"],
            "node_feature_cols": ["degree", "street_count", "dist_to_drive", "connectivity_score", "largest_component_ratio", "num_components"],
            "hidden_dim": cfg.cw_hidden,
            "seed": cfg.seed,
        },
        out_dir=cfg.artifacts_dir,
    )

    print("\n✓ Done. You can now run inference on Place with run_Place_inference.py")


if __name__ == "__main__":
    main()