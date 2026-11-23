"""Synthetic H&E-like cores for tests and smoke runs (no real data needed).

Epithelioid cores contain round nuclei; sarcomatoid cores elongated spindle
nuclei; biphasic cores contain both in separate halves.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from skimage.draw import ellipse
from skimage.measure import label as cc_label


def make_core(kind: str, size: int = 512, n_cells: int = 120, rng=None):
    rng = np.random.default_rng(rng)
    img = np.zeros((size, size, 3), dtype=np.float32)
    img[:] = (0.93, 0.72, 0.82)  # eosin-pink background
    img += rng.normal(0, 0.02, img.shape)
    labels = np.zeros((size, size), dtype=np.int32)
    types = []
    k = 0
    for _ in range(n_cells):
        r, c = rng.integers(12, size - 12, 2)
        if kind == "E":
            spindle = False
        elif kind == "S":
            spindle = True
        else:
            spindle = c > size // 2
        if spindle:
            a, b = rng.uniform(9, 14), rng.uniform(2.5, 4)
        else:
            a = b = rng.uniform(5, 7)
        rr, cc = ellipse(r, c, a, b, shape=labels.shape, rotation=rng.uniform(0, np.pi))
        if (labels[rr, cc] > 0).any():
            continue
        k += 1
        labels[rr, cc] = k
        img[rr, cc] = (0.35, 0.25, 0.55) + rng.normal(0, 0.03, 3)
        t = 0 if rng.random() < 0.8 else rng.integers(1, 5)
        types.append(t)
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return img, labels, np.asarray(types, dtype=int)


def labels_to_cellvit_json(labels: np.ndarray, types: np.ndarray, path: Path):
    """Write a label map as a CellViT-style json (for testing the reader)."""
    from skimage.measure import find_contours, regionprops

    cells = []
    for p in regionprops(labels):
        crop = np.pad(p.image.astype(float), 1)
        cs = find_contours(crop, 0.5)
        if not cs:
            continue
        cont = max(cs, key=len)
        r0, c0 = p.bbox[0] - 1, p.bbox[1] - 1
        xy = [[float(cc + c0), float(rr + r0)] for rr, cc in cont]
        cells.append({
            "contour": xy,
            "centroid": [float(p.centroid[1]), float(p.centroid[0])],
            "type": int(types[p.label - 1]) + 1,
            "type_prob": 0.9,
        })
    with open(path, "w") as f:
        json.dump({"cells": cells}, f)


def threshold_segment(image: np.ndarray, n_types: int):
    """Crude dark-blob segmentation used only by the ``dummy`` segmenter."""
    g = image.astype(float).mean(-1)
    labels = cc_label(g < 120).astype(np.int32)
    n = int(labels.max())
    return labels, np.full((n, n_types), 1.0 / n_types, dtype=np.float32)


def make_dataset(root: str | Path, n_per_class: int = 6, size: int = 384, n_cells: int = 90,
                 mpp: float = 0.5, seed: int = 0):
    """Create a toy dataset folder with images, CellViT-style jsons and a CSV."""
    import pandas as pd
    from PIL import Image

    root = Path(root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "cellvit").mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    i = 0
    for kind in "EBS":
        for j in range(n_per_class):
            img, lab, types = make_core(kind, size, n_cells, rng)
            cid = f"core{i:03d}"
            Image.fromarray(img).save(root / "images" / f"{cid}.png")
            labels_to_cellvit_json(lab, types, root / "cellvit" / f"{cid}.json")
            rows.append({
                "core_id": cid, "label": kind, "group": f"slide{j % 3}",
                "image": str(root / "images" / f"{cid}.png"), "mpp": mpp,
                "cellvit_json": str(root / "cellvit" / f"{cid}.json"), "cellvit_graph": "",
            })
            i += 1
    df = pd.DataFrame(rows)
    df.to_csv(root / "cores.csv", index=False)
    return df
