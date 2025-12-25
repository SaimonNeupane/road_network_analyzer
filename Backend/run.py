"""
Run inference on Place using saved Houten-trained models.
Produces GeoJSON with WGS84 (Lat/Lon) coordinates for React Frontend.
"""

from __future__ import annotations

import json
import sys
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import networkx as nx
import osmnx as ox
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from shapely.geometry import LineString, mapping  # Required for GeoJSON


# ----------------------------- Config ---------------------------------
@dataclass
class InferConfig:
    District, Place = sys.argv[1], sys.argv[2]
    place_target: str = f"{Place}, Nepal"
    district_name_for_csv: str = f"{District}"
    census_csv_path: str = "nepal_rural_data.csv"
    artifacts_dir: str = "artifacts"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    radius_m: float = 600.0
    max_candidates: int = 250000
    batch_size: int = 8192
    top_k: int = 25
    edge_betw_k: int = 800
    seed: int = 42


# ----------------------------- Helpers --------------------------------
def highway_to_str(val) -> str:
    if val is None:
        return "unknown"
    if isinstance(val, (list, tuple)) and len(val) > 0:
        return str(val[0])
    return str(val)


def junction_is_roundabout(val) -> int:
    return int(str(val) == "roundabout")


def approximate_edge_betweenness(
    G: nx.MultiDiGraph, k: int = 800, weight: str = "length"
) -> Dict[Tuple[int, int], float]:
    Gs = nx.Graph(G)
    if Gs.number_of_edges() == 0:
        return {}
    eb = nx.edge_betweenness_centrality(
        Gs, k=min(k, Gs.number_of_nodes()), weight=weight, seed=42
    )
    return {(min(u, v), max(u, v)): float(val) for (u, v), val in eb.items()}


def compute_connectivity_metrics(G: nx.MultiDiGraph) -> Dict[str, float]:
    Gs = nx.Graph(G)
    n = Gs.number_of_nodes()
    m = Gs.number_of_edges()
    if n == 0:
        return {
            "connectivity_score": 0.0,
            "largest_component_ratio": 0.0,
            "num_components": 0.0,
        }
    comps = list(nx.connected_components(Gs))
    lcc = max(comps, key=len)
    lcc_ratio = len(lcc) / n
    connectivity_score = (m / n) * lcc_ratio
    return {
        "connectivity_score": float(connectivity_score),
        "largest_component_ratio": float(lcc_ratio),
        "num_components": float(len(comps)),
    }


def load_pop_density(csv_path: str, district: str) -> float:
    import pandas as pd

    if not os.path.exists(csv_path):
        print(f"Warning: CSV {csv_path} not found. Using default density.")
        return 0.0
    df = pd.read_csv(csv_path)
    d = df[df["District"] == district].copy()
    if d.empty:
        print(f"Warning: District '{district}' not found. Using default density.")
        return 0.0
    d["Total population"] = pd.to_numeric(
        d["Total population"], errors="coerce"
    ).fillna(0)
    d["Total household number"] = pd.to_numeric(
        d["Total household number"], errors="coerce"
    ).fillna(0)
    return float(d["Total population"].sum()) / (
        float(d["Total household number"].sum()) + 1e-9
    )


def build_kdtree_from_graph_nodes(G: nx.MultiDiGraph):
    node_ids = []
    xy = []
    for n, d in G.nodes(data=True):
        node_ids.append(n)
        xy.append((float(d["x"]), float(d["y"])))
    xy = np.array(xy, dtype=float)
    return node_ids, xy, (cKDTree(xy) if len(xy) else None)


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
        return self.conv2(x, edge_index)


class LinkPredModel(torch.nn.Module):
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

    def forward(self, x, edge_index, edge_label_index):
        z = self.encoder(x, edge_index)
        row, col = edge_label_index
        h = torch.cat([z[row], z[col]], dim=-1)
        return self.mlp(h).squeeze(-1)


class CyclewayEdgeClassifier(torch.nn.Module):
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

    def forward(self, x, edge_index, edge_u, edge_v, edge_attr):
        z = self.encoder(x, edge_index)
        h = torch.cat([z[edge_u], z[edge_v], edge_attr], dim=-1)
        return self.mlp(h).squeeze(-1)


class GraphDataset:
    def __init__(self, G: nx.MultiDiGraph, node_features: Dict[int, Dict[str, float]]):
        self.G = nx.Graph(G)
        self.node_features = node_features
        self.node_to_idx = {n: i for i, n in enumerate(self.G.nodes())}
        self.idx_to_node = {i: n for n, i in self.node_to_idx.items()}

    def build(self) -> Data:
        nodes = list(self.G.nodes())
        cols = list(self.node_features[nodes[0]].keys()) if nodes else []
        X = torch.zeros((len(nodes), len(cols)), dtype=torch.float)
        for i, n in enumerate(nodes):
            X[i] = torch.tensor(
                [float(self.node_features[n][c]) for c in cols], dtype=torch.float
            )
        if len(nodes) > 1:
            X = (X - X.mean(dim=0)) / (X.std(dim=0) + 1e-6)
        edges = [(self.node_to_idx[u], self.node_to_idx[v]) for u, v in self.G.edges()]
        edge_index = (
            torch.tensor(edges + [(v, u) for (u, v) in edges], dtype=torch.long)
            .t()
            .contiguous()
        )
        return Data(x=X, edge_index=edge_index, num_nodes=len(nodes))


# ----------------------------- Feature Builder -------------------------
class FeatureBuilder:
    def __init__(self, G_drive: nx.MultiDiGraph, pop_density: float, betw_k: int):
        self.G_drive = G_drive
        self.pop_density = float(pop_density)
        self.drive_ids, self.drive_xy, self.drive_tree = build_kdtree_from_graph_nodes(
            G_drive
        )
        self.conn = compute_connectivity_metrics(G_drive)
        self.edge_betw = approximate_edge_betweenness(G_drive, k=betw_k)

    def node_features(self, G: nx.MultiDiGraph) -> Dict[int, Dict[str, float]]:
        feats = {}
        for node, data in G.nodes(data=True):
            deg = float(nx.Graph(G).degree(node))
            sc = float(data.get("street_count", deg))
            dist_to_drive = 1000.0
            if self.drive_tree:
                d, _ = self.drive_tree.query([float(data["x"]), float(data["y"])], k=1)
                dist_to_drive = float(d)
            feats[node] = {
                "degree": deg,
                "street_count": sc,
                "dist_to_drive": dist_to_drive,
                "connectivity_score": self.conn["connectivity_score"],
                "largest_component_ratio": self.conn["largest_component_ratio"],
                "num_components": self.conn["num_components"],
                "pop_density": self.pop_density,
            }
        return feats

    def edge_attr_for_drive_edges(
        self, G_drive: nx.MultiDiGraph, highway_vocab: Dict[str, int]
    ) -> Tuple[List, torch.Tensor]:
        edges_gdf = ox.graph_to_gdfs(G_drive, nodes=False, edges=True)
        edge_keys = list(edges_gdf.index)
        edge_attr = torch.zeros((len(edge_keys), 4), dtype=torch.float)
        max_hw = max(1, max(highway_vocab.values()))
        for i, (u, v, k) in enumerate(edge_keys):
            row = edges_gdf.loc[(u, v, k)]
            hw_idx = float(
                highway_vocab.get(highway_to_str(row.get("highway")), 0)
            ) / float(max_hw)
            a, b = (u, v) if u < v else (v, u)
            edge_attr[i] = torch.tensor(
                [
                    float(row.get("length", 0)) / 1000.0,
                    float(junction_is_roundabout(row.get("junction"))),
                    hw_idx,
                    float(self.edge_betw.get((a, b), 0)),
                ],
                dtype=torch.float,
            )
        return edge_keys, edge_attr


# ----------------------------- Candidate Gen -------------------------
def generate_radius_candidates(G_drive, dataset, radius_m, max_candidates, seed):
    rng = np.random.default_rng(seed)
    node_ids = list(dataset.node_to_idx.keys())
    xy = np.array(
        [
            (float(G_drive.nodes[n]["x"]), float(G_drive.nodes[n]["y"]))
            for n in node_ids
        ],
        dtype=float,
    )
    tree = cKDTree(xy)
    existing = set()
    for u, v in nx.Graph(G_drive).edges():
        ui, vi = dataset.node_to_idx[u], dataset.node_to_idx[v]
        existing.add(tuple(sorted((ui, vi))))

    candidates = []
    order = rng.permutation(len(node_ids))
    for idx in order:
        nbrs = tree.query_ball_point(xy[idx], r=radius_m)
        rng.shuffle(nbrs)
        for n_idx in nbrs[:50]:
            if idx == n_idx:
                continue
            pair = tuple(sorted((int(idx), int(n_idx))))
            if pair not in existing:
                candidates.append(pair)
                existing.add(pair)
            if len(candidates) >= max_candidates:
                break
        if len(candidates) >= max_candidates:
            break

    if not candidates:
        raise RuntimeError("No candidates found.")
    return torch.tensor(candidates, dtype=torch.long).t().contiguous()


# ----------------------------- Main ------------------------------------
def main():
    cfg = InferConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    print(f"Device: {cfg.device}")

    # Load Metadata & Models
    meta_lp = torch.load(
        Path(cfg.artifacts_dir) / "drive_linkpred_meta.pt", map_location="cpu"
    )
    meta_cw = torch.load(
        Path(cfg.artifacts_dir) / "cycleway_edgeclf_meta.pt", map_location="cpu"
    )

    # 1. DOWNLOAD RAW GRAPH (LAT/LON) - KEEP THIS FOR FINAL EXPORT!
    print(f"Downloading Place data for {cfg.place_target}...")
    G_raw = ox.graph_from_place(cfg.place_target, network_type="drive", simplify=True)

    # 2. CREATE PROJECTED GRAPH (METERS) - USE THIS FOR MATH/AI
    G_drive = ox.project_graph(G_raw)
    print(f"✓ Graph: {len(G_drive.nodes)} nodes, {len(G_drive.edges)} edges")

    pop_density = load_pop_density(cfg.census_csv_path, cfg.district_name_for_csv)

    # Build Features
    fb = FeatureBuilder(
        G_drive=G_drive, pop_density=pop_density, betw_k=cfg.edge_betw_k
    )
    node_feats = fb.node_features(G_drive)
    if "pop_density" not in meta_lp["node_feature_cols"]:
        for n in node_feats:
            node_feats[n].pop("pop_density", None)

    ds = GraphDataset(G_drive, node_feats)
    data = ds.build().to(cfg.device)

    # Load Models
    lp_model = LinkPredModel(data.num_node_features, int(meta_lp["hidden_dim"])).to(
        cfg.device
    )
    lp_model.load_state_dict(
        torch.load(
            Path(cfg.artifacts_dir) / "drive_linkpred_model.pt", map_location=cfg.device
        )
    )
    lp_model.eval()

    cw_model = CyclewayEdgeClassifier(
        data.num_node_features, 4, int(meta_cw["hidden_dim"])
    ).to(cfg.device)
    cw_model.load_state_dict(
        torch.load(
            Path(cfg.artifacts_dir) / "cycleway_edgeclf_model.pt",
            map_location=cfg.device,
        )
    )
    cw_model.eval()

    # --- Inference: Cycleways ---
    print("Scoring existing edges...")
    edge_keys, edge_attr = fb.edge_attr_for_drive_edges(
        G_drive, meta_lp["highway_vocab"]
    )
    u_vec = torch.tensor(
        [ds.node_to_idx[u] for u, v, k in edge_keys],
        dtype=torch.long,
        device=cfg.device,
    )
    v_vec = torch.tensor(
        [ds.node_to_idx[v] for u, v, k in edge_keys],
        dtype=torch.long,
        device=cfg.device,
    )
    edge_attr = edge_attr.to(cfg.device)

    scores = []
    with torch.no_grad():
        for i in range(0, len(edge_keys), cfg.batch_size):
            end = i + cfg.batch_size
            scores.append(
                cw_model(
                    data.x,
                    data.edge_index,
                    u_vec[i:end],
                    v_vec[i:end],
                    edge_attr[i:end],
                )
                .detach()
                .cpu()
            )
    scores = torch.cat(scores).numpy()

    top_cycle_edges = []
    for rank, idx in enumerate(np.argsort(-scores)[: cfg.top_k], 1):
        u, v, k = edge_keys[idx]
        top_cycle_edges.append(
            {"rank": rank, "u": int(u), "v": int(v), "score": float(scores[idx])}
        )

    # --- Inference: New Roads ---
    print("Generating/Scoring candidates...")
    cand_index = generate_radius_candidates(
        G_drive, ds, cfg.radius_m, cfg.max_candidates, cfg.seed
    ).to(cfg.device)
    cand_scores = []
    with torch.no_grad():
        for i in range(0, cand_index.size(1), cfg.batch_size):
            cand_scores.append(
                lp_model(data.x, data.edge_index, cand_index[:, i : i + cfg.batch_size])
                .detach()
                .cpu()
            )
    cand_scores = torch.cat(cand_scores).numpy()

    top_new_roads = []
    for rank, idx in enumerate(np.argsort(-cand_scores)[: cfg.top_k], 1):
        u = int(ds.idx_to_node[cand_index[0, idx].item()])
        v = int(ds.idx_to_node[cand_index[1, idx].item()])
        top_new_roads.append(
            {"rank": rank, "u": u, "v": v, "score": float(cand_scores[idx])}
        )

    # --- GeoJSON Export (CRITICAL FIX: Use G_raw for Lat/Lon) ---
    print("Exporting GeoJSON...")
    features = []

    # Helper to add feature
    def add_feature(item, f_type, label):
        u, v = item["u"], item["v"]
        if u in G_raw.nodes and v in G_raw.nodes:
            # Use G_raw to get Longitude (x) and Latitude (y)
            p1 = (G_raw.nodes[u]["x"], G_raw.nodes[u]["y"])
            p2 = (G_raw.nodes[v]["x"], G_raw.nodes[v]["y"])
            features.append(
                {
                    "type": "Feature",
                    "geometry": mapping(LineString([p1, p2])),
                    "properties": {
                        "type": f_type,
                        "rank": item["rank"],
                        "score": item["score"],
                        "name": f"{label} #{item['rank']}",
                    },
                }
            )

    for item in top_new_roads:
        add_feature(item, "new_road", "Proposed Road")

    for item in top_cycle_edges:
        add_feature(item, "upgrade", "Upgrade Proposal")

    out_path = Path("outputs") / "recommendations.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, indent=2)

    print(f"Saved {len(features)} features to {out_path}")


if __name__ == "__main__":
    main()
