"""Fold-specific feature normalisation (fit on training cores only).

* handcrafted / concept blocks: median imputation, robust scaling
  (median / IQR), clipping at +-``clip``
* deep blocks (fm, cvit, ctx): median imputation, standardisation, PCA to
  ``pca_components`` and unit-variance scaling of the components

The resulting ``x`` keeps block boundaries (``self.slices``) so explanations
can be reported per block.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch_geometric.data import Data


class FoldNormalizer:
    def __init__(self, cfg: dict):
        self.fc = cfg["features"]
        self.stats: dict = {}
        self.slices: dict[str, slice] = {}
        self.feature_names: list[str] = []
        self.blocks: list[str] = []

    def fit(self, graphs, seed: int = 0):
        rng = np.random.default_rng(seed)
        avail = set.intersection(*[set(g.blocks) for g in graphs])
        self.blocks = [b for b in self.fc["blocks"] if b in avail]
        if not self.blocks:
            raise ValueError("no feature blocks available")
        n_tot = sum(g.num_nodes for g in graphs)
        frac = min(1.0, self.fc["pca_fit_max_cells"] / max(n_tot, 1))
        start = 0
        for b in self.blocks:
            parts = []
            for g in graphs:
                X = g.blocks[b].numpy()
                if frac < 1:
                    X = X[rng.random(len(X)) < frac]
                parts.append(X)
            X = np.concatenate(parts).astype(np.float64)
            med = np.nanmedian(X, 0)
            med = np.where(np.isfinite(med), med, 0.0)
            X = np.where(np.isfinite(X), X, med)
            if b in self.fc["deep_blocks"]:
                mu, sd = X.mean(0), X.std(0) + 1e-6
                Z = (X - mu) / sd
                k = int(min(self.fc["pca_components"], Z.shape[1], max(Z.shape[0] - 1, 1)))
                pca = PCA(n_components=k, random_state=seed).fit(Z)
                P = pca.transform(Z)
                psd = P.std(0) + 1e-6
                self.stats[b] = dict(kind="deep", med=med, mu=mu, sd=sd, pca=pca, psd=psd)
                names = [f"{b}_pc{i}" for i in range(k)]
                width = k
            else:
                q1, q3 = np.percentile(X, [25, 75], axis=0)
                iqr = np.where(q3 - q1 > 1e-8, q3 - q1, X.std(0) + 1e-6)
                self.stats[b] = dict(kind="robust", med=med, iqr=iqr)
                names = list(getattr(graphs[0], f"names_{b}"))
                width = X.shape[1]
            self.slices[b] = slice(start, start + width)
            self.feature_names += names
            start += width
        return self

    @property
    def dim(self) -> int:
        return len(self.feature_names)

    def transform_block(self, b: str, X: np.ndarray) -> np.ndarray:
        s = self.stats[b]
        X = X.astype(np.float64)
        X = np.where(np.isfinite(X), X, s["med"])
        if s["kind"] == "deep":
            Z = (X - s["mu"]) / s["sd"]
            out = s["pca"].transform(Z) / s["psd"] if len(Z) else np.zeros((0, len(s["psd"])))
        else:
            out = (X - s["med"]) / s["iqr"]
        return np.clip(out, -self.fc["clip"], self.fc["clip"]).astype(np.float32)

    def transform(self, g) -> Data:
        x = np.concatenate([self.transform_block(b, g.blocks[b].numpy()) for b in self.blocks], 1)
        d = Data(
            x=torch.from_numpy(x), edge_index=g.edge_index, y=g.y, coords=g.coords,
            tumour_w=g.tumour_w, num_nodes=g.num_nodes,
        )
        if getattr(g, "edge_attr", None) is not None:
            d.edge_attr = g.edge_attr
        d.core_id = g.core_id
        return d
