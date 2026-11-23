"""Nucleus segmentation inputs.

MesoGraph-FM uses CellViT++ (Hörst et al., 2025) for nucleus instance
segmentation and classification. CellViT++ is run with its own CLI
(``cellvit/detect_cells.py ... --graph``) because it manages its own
checkpoints and WSI handling; this module reads its outputs.

Every backend returns a :class:`Segmentation`: an instance label map aligned
with the core image, per-instance type probabilities and (optionally) the
CellViT++ cell-token embeddings.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from skimage.draw import polygon as draw_polygon
from skimage.measure import regionprops


@dataclass
class Segmentation:
    labels: np.ndarray  # (H, W) int32, 0 = background, 1..N instances
    type_probs: np.ndarray  # (N, T) float32, row i <-> label i+1
    cvit_emb: np.ndarray | None = None  # (N, D) float32 or None

    @property
    def n(self) -> int:
        return self.type_probs.shape[0]


def _relabel(labels: np.ndarray, type_probs: np.ndarray, emb, min_area_px: float):
    """Drop tiny objects and relabel to a contiguous 1..N range."""
    props = regionprops(labels)
    keep = [p.label for p in props if p.area >= min_area_px]
    lut = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    lut[keep] = np.arange(1, len(keep) + 1, dtype=np.int32)
    idx = np.array(keep, dtype=int) - 1
    return (
        lut[labels],
        type_probs[idx] if len(idx) else type_probs[:0],
        (emb[idx] if (emb is not None and len(idx)) else (emb[:0] if emb is not None else None)),
    )


def read_cellvit_json(
    json_path: str | Path,
    shape: tuple[int, int],
    n_types: int,
    offset_xy: tuple[float, float] = (0.0, 0.0),
    xy_order: str = "xy",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterise a CellViT/CellViT++ cell json into a label map.

    Accepts the CellViT layout ``{"cells": [{"contour": [[x, y], ...],
    "type": int, "type_prob": float | list, "centroid": [x, y]}, ...]}`` or a
    bare list of such cells. Coordinates are in core-image pixels after
    subtracting ``offset_xy`` (use this when CellViT ran on the parent slide).
    PanNuke type ids are 1-based with 0 = background; they are mapped to
    0..n_types-1. If only a scalar ``type_prob`` is present, the full
    probability vector is approximated as one-hot * prob with the remainder
    spread uniformly.
    """
    with open(json_path) as f:
        data = json.load(f)
    cells = data["cells"] if isinstance(data, dict) else data
    H, W = shape
    labels = np.zeros((H, W), dtype=np.int32)
    probs, cents = [], []
    k = 0
    for c in cells:
        cont = np.asarray(c["contour"], dtype=float)
        if cont.ndim != 2 or len(cont) < 3:
            continue
        if xy_order == "yx":
            cont = cont[:, ::-1]
        xs, ys = cont[:, 0] - offset_xy[0], cont[:, 1] - offset_xy[1]
        rr, cc = draw_polygon(ys, xs, shape=(H, W))
        if len(rr) == 0:
            continue
        k += 1
        labels[rr, cc] = k
        tp = c.get("type_prob", 1.0)
        t = int(c.get("type", 1))
        t = min(max(t - 1, 0), n_types - 1) if t >= 1 else 0
        if isinstance(tp, (list, tuple)):
            v = np.asarray(tp, dtype=np.float32)
            if len(v) == n_types + 1:  # includes background
                v = v[1:]
            v = v[:n_types]
            v = v / max(v.sum(), 1e-8)
        else:
            p = float(tp)
            v = np.full(n_types, (1 - p) / max(n_types - 1, 1), dtype=np.float32)
            v[t] = p
        probs.append(v)
        cents.append([xs.mean(), ys.mean()])
    return labels, np.asarray(probs, dtype=np.float32).reshape(-1, n_types), np.asarray(cents).reshape(-1, 2)


_EMB_KEYS = ("x", "embeddings", "cell_tokens", "tokens", "features")
_POS_KEYS = ("positions", "centroids", "pos", "centroid")


def read_cellvit_graph(pt_path: str | Path, centroids_xy: np.ndarray, tol_px: float = 4.0,
                       offset_xy=(0.0, 0.0)) -> np.ndarray | None:
    """Load CellViT++ ``--graph`` output and match embeddings to our cells.

    The exact key names of the .pt file are not documented, so common names are
    tried. Embeddings are matched to cells by nearest centroid; unmatched cells
    get NaN (later imputed with the training-fold median).
    """
    import torch

    obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    get = (lambda k: getattr(obj, k, None)) if not isinstance(obj, dict) else obj.get
    emb = next((get(k) for k in _EMB_KEYS if get(k) is not None), None)
    pos = next((get(k) for k in _POS_KEYS if get(k) is not None), None)
    if emb is None or pos is None:
        raise KeyError(f"Could not find embeddings/positions in {pt_path}; keys tried {_EMB_KEYS} / {_POS_KEYS}")
    emb = np.asarray(emb, dtype=np.float32)
    pos = np.asarray(pos, dtype=np.float64) - np.asarray(offset_xy)
    out = np.full((len(centroids_xy), emb.shape[1]), np.nan, dtype=np.float32)
    if len(centroids_xy) == 0:
        return out
    d, j = cKDTree(pos).query(centroids_xy, distance_upper_bound=tol_px)
    ok = np.isfinite(d)
    out[ok] = emb[j[ok]]
    return out


def segment_core(row: dict, image: np.ndarray, cfg: dict) -> Segmentation:
    ex = cfg["extract"]
    n_types = len(ex["cell_types"])
    mpp = float(row["mpp"])
    min_area_px = ex["min_nucleus_area_um2"] / (mpp ** 2)
    H, W = image.shape[:2]
    backend = ex["segmenter"]
    emb = None
    if backend == "cellvit_files":
        off = (float(row.get("offset_x", 0) or 0), float(row.get("offset_y", 0) or 0))
        labels, probs, cents = read_cellvit_json(row["cellvit_json"], (H, W), n_types, off, ex["cellvit_xy_order"])
        gp = row.get("cellvit_graph")
        if isinstance(gp, str) and gp:
            emb = read_cellvit_graph(gp, cents + 0.0, offset_xy=off)
    elif backend == "label_map":
        labels = np.load(row["mask"]).astype(np.int32)
        tp = row.get("type_probs")
        if isinstance(tp, str) and tp:
            probs = np.load(tp).astype(np.float32)
        else:
            probs = np.full((int(labels.max()), n_types), 1.0 / n_types, dtype=np.float32)
    elif backend == "dummy":
        from .synthetic import threshold_segment

        labels, probs = threshold_segment(image, n_types)
    else:
        raise ValueError(f"unknown segmenter {backend}")
    labels, probs, emb = _relabel(labels, probs, emb, min_area_px)
    return Segmentation(labels=labels, type_probs=probs, cvit_emb=emb)
