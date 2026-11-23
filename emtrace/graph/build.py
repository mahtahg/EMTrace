"""Cell-graph construction (sparse; coordinates in µm).

Graphs keep the *raw* feature blocks (``blk_<name>``); the fold-specific
:class:`~emtrace.graph.normalize.FoldNormalizer` turns them into ``x``.
This keeps normalisation statistics out of the test fold (the original code
fitted the scaler on every core, test cores included).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data

from ..config import LABEL_MAP


def radius_edges(coords: np.ndarray, radius: float, max_nb: int):
    """Symmetric radius graph capped at ``max_nb`` nearest neighbours per node."""
    n = len(coords)
    if n < 2:
        return np.zeros((2, 0), dtype=np.int64), np.zeros(0)
    tree = cKDTree(coords)
    d, j = tree.query(coords, k=min(max_nb + 1, n), distance_upper_bound=radius)
    d, j = d.reshape(n, -1)[:, 1:], j.reshape(n, -1)[:, 1:]
    ok = np.isfinite(d)
    src = np.repeat(np.arange(n), ok.sum(1))
    dst = j[ok]
    e = np.stack([np.r_[src, dst], np.r_[dst, src]])
    e = np.unique(e, axis=1)
    e = e[:, e[0] != e[1]]
    dist = np.linalg.norm(coords[e[0]] - coords[e[1]], axis=1)
    return e.astype(np.int64), dist


def build_graph(npz_path: str | Path, label: str, core_id: str, group: str, cfg: dict) -> Data:
    z = np.load(npz_path, allow_pickle=False)
    coords = z["coords_um"].astype(np.float64)
    tp = z["type_probs"].astype(np.float32)
    gc = cfg["graph"]
    ei, dist = radius_edges(coords, gc["radius_um"], gc["max_neighbours"])
    g = Data(edge_index=torch.from_numpy(ei), num_nodes=len(coords))
    if gc["edge_features"] and ei.shape[1]:
        dxy = (coords[ei[1]] - coords[ei[0]]) / gc["radius_um"]
        hard = tp.argmax(1)
        same = (hard[ei[0]] == hard[ei[1]]).astype(np.float32)
        ea = np.c_[dist / gc["radius_um"], dxy, same]
        g.edge_attr = torch.from_numpy(ea.astype(np.float32))
    elif gc["edge_features"]:
        g.edge_attr = torch.zeros((0, 4))
    g.coords = torch.from_numpy(coords.astype(np.float32))
    g.type_probs = torch.from_numpy(tp)
    tumour_idx = cfg["extract"]["cell_types"].index(cfg["extract"]["tumour_type"])
    g.tumour_w = torch.from_numpy(tp[:, tumour_idx].copy())
    blocks = {}
    for k in z.files:
        if k.startswith("blk_"):
            name = k[4:]
            blocks[name] = torch.from_numpy(z[k].astype(np.float32))
            setattr(g, f"names_{name}", json.loads(str(z[f"names_{name}"])))
    g.blocks = blocks
    g.y = torch.tensor([LABEL_MAP[label]], dtype=torch.long)
    g.core_id = core_id
    g.group = str(group)
    return g


def build_dataset(cfg: dict, csv: str | None = None, out_dir: str | None = None):
    df = pd.read_csv(csv or cfg["paths"]["dataset_csv"]).fillna("")
    fdir = Path(cfg["paths"]["features_dir"])
    gdir = Path(out_dir or cfg["paths"]["graphs_dir"])
    gdir.mkdir(parents=True, exist_ok=True)
    graphs = []
    for r in df.to_dict("records"):
        g = build_graph(fdir / f"{r['core_id']}.npz", r["label"], r["core_id"], r.get("group", ""), cfg)
        torch.save(g, gdir / f"{r['core_id']}.pt")
        graphs.append(g)
    return graphs


def load_graphs(gdir: str | Path, core_ids=None):
    gdir = Path(gdir)
    paths = [gdir / f"{c}.pt" for c in core_ids] if core_ids is not None else sorted(gdir.glob("*.pt"))
    return [torch.load(p, weights_only=False) for p in paths]
