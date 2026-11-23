"""EMTrace model (EMTraceNet).

Architecture follows MesoGraph (Eastwood et al.): an MLP on node features,
then EdgeConv message passing, with a linear head on every layer whose
outputs are summed into per-cell [epithelioid, sarcomatoid] scores; a learned
core-conditioned threshold maps them through a sigmoid and they are pooled
into core scores. Changes: vectorised threshold (no per-graph Python loop),
optional GATv2 with edge features, optional attention pooling, no virtual
core node (the core summary is a pooled mean of the input features), and the
node encoder is a separate module so it can be self-supervised.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import EdgeConv, GATv2Conv
from torch_geometric.utils import scatter, softmax


def mlp(i, o, dropout=0.0):
    return nn.Sequential(
        nn.Linear(i, o), nn.BatchNorm1d(o), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(o, o), nn.BatchNorm1d(o), nn.ReLU(),
    )


class CellGraphEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], conv: str = "edgeconv", dropout: float = 0.0,
                 edge_dim: int | None = 4):
        super().__init__()
        self.hidden = hidden
        self.conv_type = conv
        self.first = mlp(in_dim, hidden[0], dropout)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for hin, hout in zip(hidden[:-1], hidden[1:]):
            if conv == "edgeconv":
                self.convs.append(EdgeConv(mlp(2 * hin, hout), aggr="mean"))
                self.norms.append(nn.Identity())
            elif conv == "gatv2":
                self.convs.append(GATv2Conv(hin, hout, heads=2, concat=False, edge_dim=edge_dim))
                self.norms.append(nn.Sequential(nn.BatchNorm1d(hout), nn.ReLU()))
            else:
                raise ValueError(conv)
        self.dropout = nn.Dropout(dropout)

    @property
    def out_dim(self) -> int:
        return sum(self.hidden)

    def forward(self, x, edge_index, edge_attr=None) -> list[torch.Tensor]:
        hs = [self.first(x)]
        h = hs[0]
        for conv, norm in zip(self.convs, self.norms):
            if self.conv_type == "gatv2":
                h = norm(conv(self.dropout(h), edge_index, edge_attr))
            else:
                h = conv(self.dropout(h), edge_index)
            hs.append(h)
        return hs


class EMTraceNet(nn.Module):
    def __init__(self, in_dim: int, mcfg: dict, edge_dim: int | None = 4):
        super().__init__()
        self.encoder = CellGraphEncoder(in_dim, mcfg["hidden"], mcfg["conv"], mcfg["dropout"], edge_dim)
        self.heads = nn.ModuleList([nn.Linear(h, 2) for h in mcfg["hidden"]])
        self.learnable_threshold = mcfg["learnable_threshold"]
        if self.learnable_threshold:
            self.thresh = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, 2))
        self.pooling = mcfg["pooling"]
        if self.pooling == "attention":
            self.att = nn.Sequential(nn.Linear(self.encoder.out_dim, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, x, edge_index, batch, edge_attr=None):
        """Returns dict: core [G,2], node [N,2] (post-threshold), logits [N,2], emb [N,H]."""
        hs = self.encoder(x, edge_index, edge_attr)
        Z = sum(head(h) for head, h in zip(self.heads, hs))
        n_graphs = int(batch.max()) + 1 if batch.numel() else 1
        if self.learnable_threshold:
            core_x = scatter(x, batch, dim=0, dim_size=n_graphs, reduce="mean")
            t = self.thresh(core_x)[batch]
            node = torch.sigmoid(Z - t)
        else:
            node = torch.sigmoid(Z)
        emb = torch.cat(hs, 1)
        if self.pooling == "attention":
            a = softmax(self.att(emb).squeeze(-1), batch, num_nodes=x.shape[0])
            core = scatter(node * a[:, None], batch, dim=0, dim_size=n_graphs, reduce="sum")
        else:
            core = scatter(node, batch, dim=0, dim_size=n_graphs, reduce="mean")
        return {"core": core, "node": node, "logits": Z, "emb": emb}

    @staticmethod
    def core_score(core: torch.Tensor) -> torch.Tensor:
        """Scalar per core; higher = more sarcomatoid (as in MesoGraph)."""
        return core[:, 1] - core[:, 0]
