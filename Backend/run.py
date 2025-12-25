"""
Run inference on Place using saved Houten-trained models.

Produces:
- Top-K existing drive edges to prioritize for cycleway upgrades
- Top-K new driveway candidate links (non-edges) to build

Memory-safe for 16GB RAM:
- uses radius-based candidate generation (no O(N^2))
- batches scoring on GPU (RTX 4050)
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
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


# ----------------------------- Config ---------------------------------


@dataclass
class InferConfig:
    place_target: str = "Dhulikhel, Nepal"
    district_name_for_csv: str = "Kavrepalanchok"

    census_csv_path: str = "nepal_rural_data.csv"  # your file
    artifacts_dir: str = "artifacts"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Candidate generation for new roads
    radius_m: float = 600.0         # only connect nodes within 600m
    max_candidates: int = 250000    # cap to stay safe
    batch_size: int = 8192
    top_k: int = 25

    # Betweenness approx for edge ranking (can be slower; keep moderate)
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


def approximate_edge_betweenness(G: nx.MultiDiGraph, k: int = 800, weight: str = "length") -> Dict[Tuple[int, int], float]:
    Gs = nx.Graph(G)
    if Gs.number_of_edges() == 0:
        return {}
    eb = nx.edge_betweenness_centrality(Gs, k=min(k, Gs.number_of_nodes()), weight=weight, seed=42)
    return {(min(u, v), max(u, v)): float(val) for (u, v), val in eb.items()}


def load_pop_density(csv_path: str, district: str) -> float:
    import pandas as pd

    df = pd.read_csv(csv_path)
    d = df[df["District"] == district].copy()
    if d.empty:
        raise ValueError(f"District '{district}' not found in {csv_path}")
    d["Total population"] = pd.to_numeric(d["Total population"], errors="coerce").fillna(0)
    d["Total household number"] = pd.to_numeric(d["Total household number"], errors="coerce").fillna(0)
    total_pop = float(d["Total population"].sum())
    total_house = float(d["Total household number"].sum())
    return total_pop / (total_house + 1e-9)


def build_kdtree_from_graph_nodes(G: nx.MultiDiGraph):
    node_ids = []
    xy = []
    for n, d in G.nodes(data=True):
        node_ids.append(n)
        xy.append((float(d["x"]), float(d["y"])))
    xy = np.array(xy, dtype=float)
    return node_ids, xy, (cKDTree(xy) if len(xy) else None)


# ----------------------------- PyG ------------------------------------


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


class GraphDataset:
    def __init__(self, G: nx.MultiDiGraph, node_features: Dict[int, Dict[str, float]]):
        self.G = nx.Graph(G)
        self.node_features = node_features
        self.node_to_idx: Dict[int, int] = {}
        self.idx_to_node: Dict[int, int] = {}

    def build(self) -> Data:
        nodes = list(self.G.nodes())
        self.node_to_idx = {n: i for i, n in enumerate(nodes)}
        self.idx_to_node = {i: n for n, i in self.node_to_idx.items()}

        # Use actual feature columns from the first node
        if nodes:
            cols = list(self.node_features[nodes[0]].keys())
        else:
            cols = []
        
        X = torch.zeros((len(nodes), len(cols)), dtype=torch.float)
        for i, n in enumerate(nodes):
            f = self.node_features[n]
            X[i] = torch.tensor([float(f[c]) for c in cols], dtype=torch.float)

        X = (X - X.mean(dim=0)) / (X.std(dim=0) + 1e-6)

        edges = [(self.node_to_idx[u], self.node_to_idx[v]) for u, v in self.G.edges()]
        edge_index = torch.tensor(edges + [(v, u) for (u, v) in edges], dtype=torch.long).t().contiguous()
        data = Data(x=X, edge_index=edge_index)
        data.num_nodes = len(nodes)
        return data


# ----------------------------- Feature Builder -------------------------


class FeatureBuilder:
    def __init__(self, G_drive: nx.MultiDiGraph, pop_density: float, betw_k: int):
        self.G_drive = G_drive
        self.pop_density = float(pop_density)

        self.drive_ids, self.drive_xy, self.drive_tree = build_kdtree_from_graph_nodes(G_drive)
        self.conn = compute_connectivity_metrics(G_drive)
        self.edge_betw = approximate_edge_betweenness(G_drive, k=betw_k)

    def node_features(self, G: nx.MultiDiGraph) -> Dict[int, Dict[str, float]]:
        feats = {}
        for node, data in G.nodes(data=True):
            x, y = float(data["x"]), float(data["y"])
            deg = float(nx.Graph(G).degree(node))
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
                "connectivity_score": self.conn["connectivity_score"],
                "largest_component_ratio": self.conn["largest_component_ratio"],
                "num_components": self.conn["num_components"],
                "pop_density": self.pop_density,
            }
        return feats

    def edge_attr_for_drive_edges(self, G_drive: nx.MultiDiGraph, highway_vocab: Dict[str, int]) -> Tuple[List[Tuple[int, int, int]], torch.Tensor]:
        edges_gdf = ox.graph_to_gdfs(G_drive, nodes=False, edges=True)
        edge_keys = list(edges_gdf.index)
        E = len(edge_keys)
        edge_attr = torch.zeros((E, 4), dtype=torch.float)

        max_hw = max(1, max(highway_vocab.values()))

        for i, (u, v, k) in enumerate(edge_keys):
            row = edges_gdf.loc[(u, v, k)]
            length = float(row.get("length", 0.0))
            is_roundabout = float(junction_is_roundabout(row.get("junction", None)))
            highway = highway_to_str(row.get("highway", None))
            hw_idx_norm = float(highway_vocab.get(highway, 0)) / float(max_hw)

            a, b = (u, v) if u < v else (v, u)
            betw = float(self.edge_betw.get((a, b), 0.0))
            edge_attr[i] = torch.tensor([length / 1000.0, is_roundabout, hw_idx_norm, betw], dtype=torch.float)

        return edge_keys, edge_attr


# ----------------------------- Candidate generation --------------------


def generate_radius_candidates(
    G_drive: nx.MultiDiGraph,
    dataset: GraphDataset,
    radius_m: float,
    max_candidates: int,
    seed: int,
):
    """
    Generates candidate non-edges (u_idx, v_idx) within radius using KDTree.
    Memory-safe: caps total candidates.
    """
    rng = np.random.default_rng(seed)
    node_ids = list(dataset.node_to_idx.keys())
    xy = np.array([(float(G_drive.nodes[n]["x"]), float(G_drive.nodes[n]["y"])) for n in node_ids], dtype=float)
    tree = cKDTree(xy)

    # existing edges set (undirected) in idx space
    existing = set()
    for u, v in nx.Graph(G_drive).edges():
        ui, vi = dataset.node_to_idx[u], dataset.node_to_idx[v]
        a, b = (ui, vi) if ui < vi else (vi, ui)
        existing.add((a, b))

    candidates = []
    # sample nodes to expand neighbors from (avoid worst-case explosion)
    order = rng.permutation(len(node_ids))

    for idx in order:
        u = idx
        nbrs = tree.query_ball_point(xy[u], r=radius_m)
        if not nbrs:
            continue
        rng.shuffle(nbrs)
        for v in nbrs[:50]:  # cap neighbors per node
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in existing:
                continue
            candidates.append((a, b))
            existing.add((a, b))
            if len(candidates) >= max_candidates:
                break
        if len(candidates) >= max_candidates:
            break

    if not candidates:
        raise RuntimeError("No candidates generated. Try increasing radius_m.")

    edge_index = torch.tensor(candidates, dtype=torch.long).t().contiguous()
    return edge_index


# ----------------------------- Main ------------------------------------


def main():
    cfg = InferConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print(f"Device: {cfg.device}")

    # Load metadata + models
    meta_lp = torch.load(Path(cfg.artifacts_dir) / "drive_linkpred_meta.pt", map_location="cpu")
    meta_cw = torch.load(Path(cfg.artifacts_dir) / "cycleway_edgeclf_meta.pt", map_location="cpu")
    highway_vocab = meta_lp["highway_vocab"]

    in_dim_expected = len(meta_lp["node_feature_cols"])
    # NOTE: training file didn't include pop_density; for inference we include it.
    # To keep dimensions consistent, we must mirror training features.
    # Quick fix: if training didn't include pop_density, set it aside (post-ranking).
    # Here we check and adjust.
    uses_pop_density = "pop_density" in meta_lp["node_feature_cols"]

    # Download Place drive graph
    print(f"Downloading Place data for {cfg.place_target} ...")
    G_drive = ox.graph_from_place(cfg.place_target, network_type="drive", simplify=True)
    G_drive = ox.project_graph(G_drive)
    print(f"✓ Place drive graph: {len(G_drive.nodes)} nodes, {len(G_drive.edges)} edges")

    # pop density for district
    pop_density = load_pop_density(cfg.census_csv_path, cfg.district_name_for_csv)
    print(f"pop_density({cfg.district_name_for_csv}) = {pop_density:.4f} (population per household)")

    # Build features
    fb = FeatureBuilder(G_drive=G_drive, pop_density=pop_density, betw_k=cfg.edge_betw_k)
    node_feats = fb.node_features(G_drive)

    # If training node features did NOT include pop_density, remove it for consistency
    if not uses_pop_density:
        for n in node_feats:
            node_feats[n].pop("pop_density", None)

    ds = GraphDataset(G_drive, node_feats)
    data = ds.build().to(cfg.device)

    # Load models with correct input dims
    hidden_lp = int(meta_lp["hidden_dim"])
    lp_model = LinkPredModel(in_channels=data.num_node_features, hidden=hidden_lp).to(cfg.device)
    lp_model.load_state_dict(torch.load(Path(cfg.artifacts_dir) / "drive_linkpred_model.pt", map_location=cfg.device))
    lp_model.eval()

    hidden_cw = int(meta_cw["hidden_dim"])
    cw_model = CyclewayEdgeClassifier(in_channels=data.num_node_features, edge_attr_dim=4, hidden=hidden_cw).to(cfg.device)
    cw_model.load_state_dict(torch.load(Path(cfg.artifacts_dir) / "cycleway_edgeclf_model.pt", map_location=cfg.device))
    cw_model.eval()

    # -------------------- (1) Cycleway upgrades on existing edges -------
    print("\nScoring existing drive edges for cycleway upgrades...")
    edge_keys, edge_attr = fb.edge_attr_for_drive_edges(G_drive, highway_vocab)

    edge_u = torch.tensor([ds.node_to_idx[u] for (u, v, k) in edge_keys], dtype=torch.long, device=cfg.device)
    edge_v = torch.tensor([ds.node_to_idx[v] for (u, v, k) in edge_keys], dtype=torch.long, device=cfg.device)
    edge_attr = edge_attr.to(cfg.device)

    # batch score edges
    scores = []
    bs = cfg.batch_size
    with torch.no_grad():
        for i in range(0, len(edge_keys), bs):
            s = cw_model(data.x, data.edge_index, edge_u[i:i+bs], edge_v[i:i+bs], edge_attr[i:i+bs])
            scores.append(s.detach().cpu())
    scores = torch.cat(scores).numpy()

    top_idx = np.argsort(-scores)[: cfg.top_k]
    top_cycle_edges = []
    for rank, i in enumerate(top_idx, start=1):
        (u, v, k) = edge_keys[i]
        top_cycle_edges.append({"rank": rank, "u": int(u), "v": int(v), "key": int(k), "score": float(scores[i])})

    print("\nTop cycleway-upgrade edges (existing roads):")
    for e in top_cycle_edges:
        print(f"  #{e['rank']}: ({e['u']}, {e['v']}, key={e['key']}) score={e['score']:.3f}")

    # -------------------- (2) New driveway links (non-edges) ------------
    print("\nGenerating candidate new driveway links (radius-based)...")
    cand_edge_index = generate_radius_candidates(
        G_drive=G_drive,
        dataset=ds,
        radius_m=cfg.radius_m,
        max_candidates=cfg.max_candidates,
        seed=cfg.seed,
    ).to(cfg.device)

    print(f"Candidates generated: {cand_edge_index.size(1)}")

    print("Scoring candidates in batches...")
    cand_scores = []
    with torch.no_grad():
        for i in range(0, cand_edge_index.size(1), bs):
            batch = cand_edge_index[:, i:i+bs]
            cand_scores.append(lp_model(data.x, data.edge_index, batch).detach().cpu())
    cand_scores = torch.cat(cand_scores).numpy()

    top_idx = np.argsort(-cand_scores)[: cfg.top_k]
    top_new_roads = []
    for rank, j in enumerate(top_idx, start=1):
        u_idx = int(cand_edge_index[0, j].item())
        v_idx = int(cand_edge_index[1, j].item())
        u = int(ds.idx_to_node[u_idx])
        v = int(ds.idx_to_node[v_idx])
        top_new_roads.append({"rank": rank, "u": u, "v": v, "score": float(cand_scores[j])})

    print("\nTop new driveway links (build new roads):")
    for e in top_new_roads:
        print(f"  #{e['rank']}: {e['u']} ↔ {e['v']} score={e['score']:.3f}")

    # Save outputs
    out = {
        "place_target": cfg.place_target,
        "district": cfg.district_name_for_csv,
        "pop_density": pop_density,
        "top_cycleway_upgrades": top_cycle_edges,
        "top_new_driveways": top_new_roads,
        "params": {
            "radius_m": cfg.radius_m,
            "max_candidates": cfg.max_candidates,
            "top_k": cfg.top_k,
        },
    }
    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/recommendations.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\nSaved outputs/recommendations.json")


if __name__ == "__main__":
    main()