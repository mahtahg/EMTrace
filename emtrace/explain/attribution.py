"""Integrated-gradients attributions of the core score to node features.

Replaces GNNExplainer. Attributions are returned per feature and aggregated
per feature block (morph, int, tex, type, nbr, fm, cvit, ctx, concept), since
single principal components of a foundation-model embedding are not
individually meaningful but block contributions are.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..models.emtrace_net import EMTraceNet


def integrated_gradients(model: EMTraceNet, d, steps: int = 32, device="cpu") -> np.ndarray:
    """Return (N, F) attributions for one graph; baseline = zeros (= training median)."""
    model.eval()
    d = d.to(device)
    x = d.x
    batch = torch.zeros(x.shape[0], dtype=torch.long, device=device)
    ea = getattr(d, "edge_attr", None)
    total = torch.zeros_like(x)
    for a in torch.linspace(1.0 / steps, 1.0, steps, device=device):
        xi = (a * x).detach().requires_grad_(True)
        out = model(xi, d.edge_index, batch, ea)
        s = EMTraceNet.core_score(out["core"]).sum()
        (g,) = torch.autograd.grad(s, xi)
        total += g
    return (x * total / steps).detach().cpu().numpy()


def explain_cores(model, graphs, normalizer, device="cpu", steps=32) -> tuple[pd.DataFrame, pd.DataFrame]:
    feat_rows, block_rows = [], []
    for d in graphs:
        A = integrated_gradients(model, d, steps, device)
        per_feat = A.sum(0)
        feat_rows.append(pd.Series(per_feat, index=normalizer.feature_names, name=d.core_id))
        block_rows.append(pd.Series({b: per_feat[s].sum() for b, s in normalizer.slices.items()}, name=d.core_id))
    return pd.DataFrame(feat_rows), pd.DataFrame(block_rows)
