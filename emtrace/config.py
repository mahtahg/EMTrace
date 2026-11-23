"""Configuration handling.

A config is a nested dict. ``DEFAULTS`` holds every option with its default;
a user YAML only needs to override what differs. Access with ``cfg["a"]["b"]``.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

LABEL_MAP = {"E": 0, "B": 1, "S": 2}

DEFAULTS: dict[str, Any] = {
    "seed": 0,
    "device": "auto",  # auto | cpu | cuda
    "paths": {
        "dataset_csv": "data/cores.csv",  # core_id,label,group,image,mpp[,cellvit_json,cellvit_graph,mask]
        "work_dir": "runs/default",
        "features_dir": None,  # default: <work_dir>/features
        "graphs_dir": None,  # default: <work_dir>/graphs
        "external_csv": None,  # optional external test cohort (same columns)
    },
    "extract": {
        "segmenter": "cellvit_files",  # cellvit_files | label_map | dummy
        "cellvit_xy_order": "xy",  # coordinate order inside CellViT json: xy | yx
        "cell_types": ["neoplastic", "inflammatory", "connective", "dead", "epithelial"],
        "tumour_type": "neoplastic",
        "min_nucleus_area_um2": 8.0,
        "encoder": {
            "name": "hoptimus1",  # hoptimus1 | dummy | none
            "mpp": 0.5,
            "tile_size": 224,
            "stride": 112,  # overlapping tiles; token features are averaged by overlap
            "batch_size": 16,
            "use_tile_context": True,  # adds tile CLS ("ctx") block per cell
            "hf_token_env": "HF_TOKEN",
        },
        "handcrafted": {
            "morphology": True,
            "fourier_harmonics": 10,
            "intensity": True,
            "ring_um": 3.0,  # cytoplasm-proxy ring width around each nucleus
            "texture": True,
            "texture_circle_um": 50.0,  # matches original QuPath 50 µm circle
            "texture_levels": 32,
            "neighbourhood": True,
            "neighbourhood_radii_um": [25.0, 50.0, 100.0],
        },
        "concepts": {  # Novelty F
            "enabled": True,
            "backend": "conch",  # conch | dummy
            "crop_um": 32.0,
            "batch_size": 64,
            "hf_token_env": "HF_TOKEN",
            "prompts": {
                "epithelioid": ["epithelioid tumour cells", "polygonal epithelioid cells with abundant eosinophilic cytoplasm"],
                "spindle": ["spindle-shaped tumour cells", "elongated spindle cells in fascicles"],
                "pleomorphic": ["pleomorphic nuclei", "markedly atypical pleomorphic tumour cells"],
                "nucleolus": ["prominent nucleoli", "nuclei with conspicuous nucleoli"],
                "mitosis": ["mitotic figure", "dividing tumour cell"],
                "necrosis": ["tumour necrosis", "necrotic debris"],
                "inflammation": ["lymphocytes", "inflammatory infiltrate"],
                "fibrous_stroma": ["dense fibrous stroma", "collagenous stroma"],
                "myxoid_stroma": ["myxoid stroma", "loose myxoid background"],
                "tubulopapillary": ["tubulopapillary architecture", "tumour cells forming tubules and papillae"],
            },
        },
    },
    "graph": {
        "radius_um": 30.0,
        "max_neighbours": 16,
        "edge_features": True,
    },
    "features": {
        # blocks present in the extracted files; any missing block is skipped
        "blocks": ["morph", "int", "tex", "type", "nbr", "fm", "cvit", "ctx", "concept"],
        "deep_blocks": ["fm", "cvit", "ctx"],  # compressed with PCA fitted on the training fold
        "pca_components": 128,
        "pca_fit_max_cells": 200000,
        "clip": 5.0,
    },
    "model": {
        "conv": "edgeconv",  # edgeconv | gatv2
        "hidden": [64, 32, 32],
        "dropout": 0.2,
        "pooling": "mean",  # mean | attention
        "learnable_threshold": True,
    },
    "pretrain": {  # Novelty G
        "enabled": True,
        "epochs": 50,
        "mask_ratio": 0.5,
        "lr": 1e-3,
        "gamma": 2.0,  # scaled-cosine exponent (GraphMAE)
        "extra_unlabelled_graphs": None,  # optional dir of extra unlabelled .pt graphs
    },
    "loss": {
        "ranking_margin": 0.8,
        "proportion": {  # Novelty A
            "enabled": True,
            "weight": 1.0,
            "threshold": 0.10,  # WHO: each component >=10% for biphasic
            "temperature": 1.0,
            "tumour_weighted": True,
            "sharpness_weight": 0.05,
            "smoothness_weight": 0.05,
        },
    },
    "train": {
        "epochs": 200,
        "lr": 5e-4,
        "weight_decay": 0.02,
        "batch_size": 16,
        "val_fraction": 0.25,
        "split": "group",  # group (leave-one-group-out) | kfold | external
        "n_folds": 5,
        "repeats": 1,
        "log_every": 20,
        "save_node_preds": True,
        "bootstrap": 1000,
    },
    "explain": {
        "methods": ["ig", "gnnexplainer"],  # integrated gradients and/or GNNExplainer
        "ig_steps": 32,
        "gnnexplainer_epochs": 200,
        "save_cell_maps": True,
    },
    "pseudotime": {  # Novelty C
        "enabled": True,
        "pca_dim": 10,
        "n_centroids": 50,
        "fit_max_cells": 50000,
        "bins": 10,
    },
}


def deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict) and k != "prompts":
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        with open(path) as f:
            deep_update(cfg, yaml.safe_load(f) or {})
    if overrides:
        deep_update(cfg, overrides)
    wd = Path(cfg["paths"]["work_dir"])
    cfg["paths"]["features_dir"] = cfg["paths"]["features_dir"] or str(wd / "features")
    cfg["paths"]["graphs_dir"] = cfg["paths"]["graphs_dir"] or str(wd / "graphs")
    return cfg


def resolve_device(cfg: dict):
    import torch

    d = cfg.get("device", "auto")
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)
