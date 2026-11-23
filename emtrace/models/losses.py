"""Training losses.

* :func:`pairwise_ranking_loss` - MesoGraph's pairwise hinge ranking loss
  (E < B < S), vectorised; the same margin is used in training and evaluation
  (the original used 0.8 vs 1.0).
* :func:`proportion_loss` - WHO-proportion constraint. Each cell gets a soft sarcomatoid
  assignment q = sigmoid((s_S - s_E) / T). The core's sarcomatoid fraction f
  is the tumour-cell-weighted mean of q (WHO proportions refer to tumour
  cells). With threshold tau (WHO: 10%) the loss enforces
  E: f <= tau, S: f >= 1 - tau, B: tau <= f <= 1 - tau,
  plus a sharpness term pushing q towards {0,1} (so f counts cells) and a
  graph-smoothness term (total variation over edges) so components form
  contiguous regions rather than scattered single cells.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.utils import scatter


def pairwise_ranking_loss(core: torch.Tensor, y: torch.Tensor, margin: float = 0.8) -> torch.Tensor:
    """core: [G,2] (E score, S score); y: [G] ordinal labels."""
    dy = torch.sign(y[None, :] - y[:, None]).float()  # dy[i,j] = sign(y_j - y_i)
    iu = torch.triu_indices(len(y), len(y), 1, device=y.device)
    dy = dy[iu[0], iu[1]]
    keep = dy != 0
    if keep.sum() == 0:
        return core.sum() * 0.0
    dz = core[iu[0]] - core[iu[1]]  # [P,2]
    target = torch.stack([dy, -dy], 1)  # E score should rank opposite to S score
    h = torch.clamp(margin - target * dz, min=0).mean(1)
    return h[keep].mean()


def sarcomatoid_assignment(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    return torch.sigmoid((logits[:, 1] - logits[:, 0]) / temperature)


def sarcomatoid_fraction(q, batch, weights=None, n_graphs=None):
    w = torch.ones_like(q) if weights is None else weights.clamp(min=0) + 1e-6
    num = scatter(q * w, batch, dim=0, dim_size=n_graphs, reduce="sum")
    den = scatter(w, batch, dim=0, dim_size=n_graphs, reduce="sum")
    return num / den.clamp(min=1e-6)


def proportion_loss(logits, batch, y, edge_index, pcfg: dict, tumour_w=None):
    n_graphs = y.shape[0]
    q = sarcomatoid_assignment(logits, pcfg["temperature"])
    w = tumour_w if (pcfg["tumour_weighted"] and tumour_w is not None) else None
    f = sarcomatoid_fraction(q, batch, w, n_graphs)
    tau = pcfg["threshold"]
    lo = torch.where(y == 2, 1 - tau, torch.where(y == 1, tau, 0.0)).float()
    hi = torch.where(y == 0, tau, torch.where(y == 1, 1 - tau, 1.0)).float()
    bound = (F.relu(lo - f) ** 2 + F.relu(f - hi) ** 2).mean()
    sharp = (q * (1 - q)).mean()
    if edge_index.numel():
        smooth = (q[edge_index[0]] - q[edge_index[1]]).abs().mean()
    else:
        smooth = q.sum() * 0.0
    total = bound + pcfg["sharpness_weight"] * sharp + pcfg["smoothness_weight"] * smooth
    return total, {"bound": bound.item(), "sharp": sharp.item(), "smooth": smooth.item()}, f
