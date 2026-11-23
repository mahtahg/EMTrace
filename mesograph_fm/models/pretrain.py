"""Novelty G: self-supervised masked-feature pretraining of the cell encoder.

GraphMAE-style (Hou et al., 2022): a random subset of nodes has its features
replaced by a learnable mask token; the encoder embeds the graph; masked
embeddings are re-masked and a one-layer graph decoder reconstructs the
original features; loss is the scaled cosine error on masked nodes. The
pretrained encoder initialises the classifier. Uses only cores from the
training fold (plus optional extra unlabelled graphs) - no labels.
Toggle with ``pretrain.enabled``.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import EdgeConv

from .mesograph import CellGraphEncoder

log = logging.getLogger(__name__)


class MaskedFeatureAutoencoder(nn.Module):
    def __init__(self, encoder: CellGraphEncoder, in_dim: int, gamma: float = 2.0):
        super().__init__()
        self.encoder = encoder
        self.mask_token = nn.Parameter(torch.zeros(1, in_dim))
        dh = encoder.hidden[0]
        self.enc2dec = nn.Linear(encoder.out_dim, dh, bias=False)
        self.decoder = EdgeConv(nn.Sequential(nn.Linear(2 * dh, dh), nn.ReLU(), nn.Linear(dh, in_dim)), aggr="mean")
        self.gamma = gamma

    def forward(self, x, edge_index, edge_attr, mask):
        xm = x.clone()
        xm[mask] = self.mask_token.to(x.dtype)
        h = torch.cat(self.encoder(xm, edge_index, edge_attr), 1)
        h = self.enc2dec(h)
        h = h.masked_fill(mask[:, None], 0.0)
        rec = self.decoder(h, edge_index)
        cos = F.cosine_similarity(rec[mask], x[mask], dim=-1)
        return ((1 - cos).clamp(min=0) ** self.gamma).mean()


def pretrain_encoder(encoder: CellGraphEncoder, graphs, in_dim: int, pcfg: dict, device, seed: int = 0,
                     batch_size: int = 8) -> list[float]:
    torch.manual_seed(seed)
    model = MaskedFeatureAutoencoder(encoder, in_dim, pcfg["gamma"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=pcfg["lr"], weight_decay=1e-4)
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)
    hist = []
    model.train()
    for ep in range(pcfg["epochs"]):
        tot, nb = 0.0, 0
        for d in loader:
            d = d.to(device)
            if d.num_nodes < 4:
                continue
            mask = torch.rand(d.num_nodes, device=device) < pcfg["mask_ratio"]
            if mask.sum() < 2:
                continue
            loss = model(d.x, d.edge_index, getattr(d, "edge_attr", None), mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        hist.append(tot / max(nb, 1))
    log.info("pretrain: loss %.4f -> %.4f", hist[0] if hist else float("nan"), hist[-1] if hist else float("nan"))
    return hist
