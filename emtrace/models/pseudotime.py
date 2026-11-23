"""Epithelioid -> sarcomatoid cell pseudotime.

A DDRTree-style principal tree is fitted to the learned cell embeddings of
training-fold tumour cells: PCA -> k-means centroids -> minimum spanning tree
over centroids. The root is the centroid whose cells have the lowest mean
sarcomatoid assignment q (most epithelioid). Each cell is projected onto its
nearest tree edge; its pseudotime is the geodesic distance from the root
along the tree, scaled to [0, 1]. Cores are summarised by the histogram,
mean, median and bimodality coefficient of their tumour cells' pseudotimes,
which describes the E->S continuum (and biphasic mixtures) rather than three
discrete classes. Fitted on the training fold only and applied to test cores.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import dijkstra, minimum_spanning_tree
from scipy.spatial.distance import cdist
from scipy.stats import kurtosis, skew
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class PrincipalTreePseudotime:
    def __init__(self, pca_dim: int = 10, n_centroids: int = 50, seed: int = 0):
        self.pca_dim, self.k, self.seed = pca_dim, n_centroids, seed

    def fit(self, emb: np.ndarray, q: np.ndarray):
        d = int(min(self.pca_dim, emb.shape[1], max(len(emb) - 1, 1)))
        self.pca = PCA(n_components=d, random_state=self.seed).fit(emb)
        P = self.pca.transform(emb)
        k = int(min(self.k, max(len(P) // 5, 2)))
        km = KMeans(n_clusters=k, n_init=4, random_state=self.seed).fit(P)
        self.C = km.cluster_centers_
        D = cdist(self.C, self.C)
        T = minimum_spanning_tree(D).toarray()
        T = T + T.T
        self.edges = np.array(np.nonzero(np.triu(T))).T  # (E,2)
        qc = np.array([q[km.labels_ == i].mean() if (km.labels_ == i).any() else np.inf for i in range(k)])
        self.root = int(np.argmin(qc))
        self.geo = dijkstra(T, indices=self.root)  # distance root -> centroid
        self.scale = float(np.max(self.geo[np.isfinite(self.geo)])) or 1.0
        return self

    def transform(self, emb: np.ndarray, chunk: int = 20000) -> np.ndarray:
        P = self.pca.transform(emb)
        a, b = self.C[self.edges[:, 0]], self.C[self.edges[:, 1]]
        ab = b - a
        L2 = (ab ** 2).sum(1) + 1e-12
        out = np.empty(len(P))
        for s in range(0, len(P), chunk):
            X = P[s:s + chunk]
            t = np.clip(((X[:, None, :] - a[None]) * ab[None]).sum(-1) / L2[None], 0, 1)  # (n,E)
            proj = a[None] + t[..., None] * ab[None]
            dist = ((X[:, None, :] - proj) ** 2).sum(-1)
            e = dist.argmin(1)
            te = t[np.arange(len(X)), e]
            ga, gb = self.geo[self.edges[e, 0]], self.geo[self.edges[e, 1]]
            out[s:s + chunk] = ga + te * (gb - ga)
        return np.clip(out / self.scale, 0, 1)


def core_summary(pt: np.ndarray, w: np.ndarray | None = None, bins: int = 10) -> dict:
    if len(pt) == 0:
        return {}
    w = np.ones_like(pt) if w is None else np.clip(w, 0, None) + 1e-6
    hist, _ = np.histogram(pt, bins=bins, range=(0, 1), weights=w)
    hist = hist / max(hist.sum(), 1e-9)
    mean = float(np.average(pt, weights=w))
    order = np.argsort(pt)
    cw = np.cumsum(w[order]) / w.sum()
    median = float(pt[order][np.searchsorted(cw, 0.5)])
    n = len(pt)
    if n > 3:
        g, k = skew(pt), kurtosis(pt)
        bc = float((g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))
    else:
        bc = float("nan")
    out = {"pt_mean": mean, "pt_median": median, "pt_bimodality": bc}
    out.update({f"pt_hist{i}": float(h) for i, h in enumerate(hist)})
    return out
