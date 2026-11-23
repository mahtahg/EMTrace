"""Explanations of the core score: integrated gradients and GNNExplainer.

Two complementary methods are provided (``explain.methods`` in the config):

* ``ig`` - integrated gradients (gradient-based, signed). Shows how much
  each node feature pushes the core score towards sarcomatoid (+) or
  epithelioid (-).
* ``gnnexplainer`` - GNNExplainer (Ying et al., 2019; the method used in
  the original MesoGraph), via PyTorch Geometric's ``Explainer``. It learns
  soft masks over node features and edges that preserve the core score.
  It gives unsigned importances, and also an **edge mask**, i.e. which cell-cell
  interactions matter, which IG does not provide. The edge mask is turned into
  a per-cell importance for spatial maps.

Both are reported per feature and aggregated per feature block (IG: sums,
since IG is additive; GNNExplainer: means of its 0-1 masks) (morph, int,
tex, type, nbr, fm, cvit, ctx, concept). Single principal components of a
foundation-model embedding mean little on their own, but block contributions
do.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.utils import scatter

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


class _CoreScore(nn.Module):
    """Adapter so PyG's Explainer sees forward(x, edge_index, **kw) -> [G, 1]."""

    def __init__(self, model: EMTraceNet):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, batch=None, edge_attr=None):
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        out = self.model(x, edge_index, batch, edge_attr)
        return EMTraceNet.core_score(out["core"]).unsqueeze(-1)


def gnn_explainer(model: EMTraceNet, d, epochs: int = 200, lr: float = 0.01, device="cpu", seed: int = 0):
    """Return (node_feat_mask [N,F], edge_mask [E], node_importance [N]) for one graph."""
    from torch_geometric.explain import Explainer, GNNExplainer

    torch.manual_seed(seed)
    model.eval()
    d = d.to(device)
    explainer = Explainer(
        model=_CoreScore(model),
        algorithm=GNNExplainer(epochs=epochs, lr=lr),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(mode="regression", task_level="graph", return_type="raw"),
    )
    batch = torch.zeros(d.num_nodes, dtype=torch.long, device=device)
    exp = explainer(d.x, d.edge_index, batch=batch, edge_attr=getattr(d, "edge_attr", None))
    nm = exp.node_mask.detach().cpu().numpy()
    em = exp.edge_mask.detach() if exp.get("edge_mask") is not None else torch.zeros(d.edge_index.shape[1])
    if em.numel():
        # cell importance = mean mask of its incident edges (as in MesoGraph's node_imp)
        ni = scatter(em.cpu(), d.edge_index[0].cpu(), dim=0, dim_size=d.num_nodes, reduce="mean").numpy()
    else:
        ni = np.zeros(d.num_nodes, dtype=np.float32)
    return nm, em.cpu().numpy(), ni


def explain_cores(model, graphs, normalizer, device="cpu", steps=32, methods=("ig",),
                  gnnexp_epochs: int = 200, node_out_dir=None) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Return {method: (per-feature df [cores x features], per-block df [cores x blocks])}.

    If ``node_out_dir`` is given, per-cell maps are saved there as
    ``<core_id>.npz`` (coords, ig_node, gnnexp_node_feat, gnnexp_cell_importance,
    gnnexp_edge_mask, edge_index) for overlaying on the core image.
    """
    res = {m: ([], []) for m in methods}
    for d in graphs:
        save = {"coords": d.coords.numpy(), "edge_index": d.edge_index.numpy()}
        for m in methods:
            if m == "ig":
                A = integrated_gradients(model, d, steps, device)
                save["ig_node"] = A.sum(1)
            elif m == "gnnexplainer":
                A, em, ni = gnn_explainer(model, d, gnnexp_epochs, device=device)
                save.update(gnnexp_node_feat=A.sum(1), gnnexp_cell_importance=ni, gnnexp_edge_mask=em)
            else:
                raise ValueError(f"unknown explanation method {m}")
            # IG is additive (completeness), so sum over cells and over a block's features.
            # GNNExplainer masks are unsigned 0-1 importances, so average instead; otherwise
            # wide blocks and large cores would dominate.
            agg = np.sum if m == "ig" else np.mean
            per_feat = agg(A, axis=0)
            res[m][0].append(pd.Series(per_feat, index=normalizer.feature_names, name=d.core_id))
            res[m][1].append(pd.Series({b: agg(per_feat[s]) for b, s in normalizer.slices.items()}, name=d.core_id))
        if node_out_dir is not None:
            np.savez_compressed(f"{node_out_dir}/{d.core_id}.npz", **save)
    return {m: (pd.DataFrame(f), pd.DataFrame(b)) for m, (f, b) in res.items()}
