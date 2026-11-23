"""Cross-validated training, prediction and pseudotime fitting."""
from __future__ import annotations

import copy
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, StratifiedKFold, train_test_split
from torch_geometric.loader import DataLoader

from ..config import resolve_device
from ..graph.build import load_graphs
from ..graph.normalize import FoldNormalizer
from ..models.losses import pairwise_ranking_loss, proportion_loss, sarcomatoid_assignment, sarcomatoid_fraction
from ..models.mesograph import MesoGraphFM
from ..models.pretrain import pretrain_encoder
from ..models.pseudotime import PrincipalTreePseudotime, core_summary
from .metrics import bootstrap_ci, core_metrics, selection_metric

log = logging.getLogger(__name__)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def stratified_batches(y: np.ndarray, batch_size: int, rng) -> list[np.ndarray]:
    """Batches with a balanced class mix (ranking loss needs mixed labels)."""
    n_b = max(len(y) // batch_size, 1)
    if n_b == 1 or np.bincount(y).min() < n_b:
        idx = rng.permutation(len(y))
        return [idx[i:i + batch_size] for i in range(0, len(y), batch_size)]
    skf = StratifiedKFold(n_splits=n_b, shuffle=True, random_state=int(rng.integers(1 << 31)))
    return [te for _, te in skf.split(np.zeros(len(y)), y)]


def make_splits(y, groups, tcfg, seed):
    y, groups = np.asarray(y), np.asarray(groups)
    if tcfg["split"] == "group":
        return list(LeaveOneGroupOut().split(y, y, groups))
    if tcfg["split"] == "kfold":
        if len(np.unique(groups)) >= tcfg["n_folds"]:
            sk = StratifiedGroupKFold(tcfg["n_folds"], shuffle=True, random_state=seed)
            return list(sk.split(y, y, groups))
        return list(StratifiedKFold(tcfg["n_folds"], shuffle=True, random_state=seed).split(y, y))
    raise ValueError(tcfg["split"])


def train_val_split(idx, y, groups, frac, seed):
    idx, y, groups = np.asarray(idx), np.asarray(y)[idx], np.asarray(groups)[idx]
    n_splits = max(int(round(1 / frac)), 2)
    if len(np.unique(groups)) >= n_splits:
        try:
            tr, va = next(StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(idx, y, groups))
            if len(np.unique(y[va])) > 1:
                return idx[tr], idx[va]
        except ValueError:
            pass
    tr, va = train_test_split(np.arange(len(idx)), test_size=frac, stratify=y, random_state=seed)
    return idx[tr], idx[va]


class Trainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = resolve_device(cfg)

    # ---------- one forward over a loader ----------
    @torch.no_grad()
    def predict(self, model, data, keep_nodes=False):
        model.eval()
        rows, nodes = [], {}
        for d in DataLoader(data, batch_size=8):
            d = d.to(self.device)
            out = model(d.x, d.edge_index, d.batch, getattr(d, "edge_attr", None))
            q = sarcomatoid_assignment(out["logits"], self.cfg["loss"]["proportion"]["temperature"])
            w = d.tumour_w if self.cfg["loss"]["proportion"]["tumour_weighted"] else None
            f = sarcomatoid_fraction(q, d.batch, w, d.num_graphs)
            score = MesoGraphFM.core_score(out["core"])
            for gi in range(d.num_graphs):
                rows.append({
                    "core_id": d.core_id[gi], "y": int(d.y[gi]),
                    "score": float(score[gi]), "core_E": float(out["core"][gi, 0]),
                    "core_S": float(out["core"][gi, 1]), "sarcomatoid_pct": 100 * float(f[gi]),
                })
                if keep_nodes:
                    m = d.batch == gi
                    nodes[d.core_id[gi]] = {
                        "coords": d.coords[m].cpu().numpy(), "node": out["node"][m].cpu().numpy(),
                        "q": q[m].cpu().numpy(), "emb": out["emb"][m].cpu().numpy(),
                        "tumour_w": d.tumour_w[m].cpu().numpy(),
                    }
        return pd.DataFrame(rows), nodes

    def _loss(self, model, d):
        out = model(d.x, d.edge_index, d.batch, getattr(d, "edge_attr", None))
        y = d.y
        loss = pairwise_ranking_loss(out["core"], y, self.cfg["loss"]["ranking_margin"])
        pc = self.cfg["loss"]["proportion"]
        if pc["enabled"]:
            lp, _, _ = proportion_loss(out["logits"], d.batch, y, d.edge_index, pc, d.tumour_w)
            loss = loss + pc["weight"] * lp
        return loss

    def fit_fold(self, train, val, in_dim, seed, extra_unlabelled=None):
        cfg = self.cfg
        set_seed(seed)
        edge_dim = train[0].edge_attr.shape[1] if getattr(train[0], "edge_attr", None) is not None else None
        model = MesoGraphFM(in_dim, cfg["model"], edge_dim).to(self.device)
        pre_hist = []
        if cfg["pretrain"]["enabled"]:
            pre_hist = pretrain_encoder(model.encoder, list(train) + list(extra_unlabelled or []), in_dim,
                                        cfg["pretrain"], self.device, seed)
        tc = cfg["train"]
        opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc["epochs"])
        ytr = np.array([int(g.y) for g in train])
        rng = np.random.default_rng(seed)
        best, best_state, hist = -np.inf, None, []
        for ep in range(1, tc["epochs"] + 1):
            model.train()
            tot = 0.0
            for b in stratified_batches(ytr, tc["batch_size"], rng):
                if len(b) < 2:
                    continue
                d = next(iter(DataLoader([train[i] for i in b], batch_size=len(b)))).to(self.device)
                loss = self._loss(model, d)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item()
            sched.step()
            vdf, _ = self.predict(model, val)
            sel = selection_metric(vdf.y.values, vdf.score.values)
            if not np.isfinite(sel):
                sel = -tot
            hist.append({"epoch": ep, "train_loss": tot, "val_sel": sel})
            if sel >= best:
                best, best_state = sel, copy.deepcopy(model.state_dict())
            if ep % tc["log_every"] == 0:
                log.info("epoch %d loss %.4f val %.4f", ep, tot, sel)
        model.load_state_dict(best_state)
        return model, {"pretrain": pre_hist, "train": hist, "best_val": best}

    def fit_pseudotime(self, model, train):
        pc = self.cfg["pseudotime"]
        _, nodes = self.predict(model, train, keep_nodes=True)
        E = np.concatenate([v["emb"] for v in nodes.values()])
        Q = np.concatenate([v["q"] for v in nodes.values()])
        W = np.concatenate([v["tumour_w"] for v in nodes.values()])
        keep = W >= 0.5 if (W >= 0.5).sum() >= 20 else np.ones(len(W), bool)
        E, Q = E[keep], Q[keep]
        rng = np.random.default_rng(self.cfg["seed"])
        if len(E) > pc["fit_max_cells"]:
            i = rng.choice(len(E), pc["fit_max_cells"], replace=False)
            E, Q = E[i], Q[i]
        return PrincipalTreePseudotime(pc["pca_dim"], pc["n_centroids"], self.cfg["seed"]).fit(E, Q)

    def run(self, graphs=None, external=None):
        cfg = self.cfg
        wd = Path(cfg["paths"]["work_dir"])
        wd.mkdir(parents=True, exist_ok=True)
        graphs = graphs if graphs is not None else load_graphs(cfg["paths"]["graphs_dir"])
        y = np.array([int(g.y) for g in graphs])
        groups = np.array([g.group for g in graphs])
        extra = None
        if cfg["pretrain"]["extra_unlabelled_graphs"]:
            extra = load_graphs(cfg["pretrain"]["extra_unlabelled_graphs"])
        all_preds, summary = [], {}
        for rep in range(cfg["train"]["repeats"]):
            seed = cfg["seed"] + rep
            if cfg["train"]["split"] == "external":
                assert external is not None, "external split needs external graphs"
                splits = [(np.arange(len(graphs)), None)]
            else:
                splits = make_splits(y, groups, cfg["train"], seed)
            for fold, (tr_idx, te_idx) in enumerate(splits):
                fdir = wd / f"rep{rep}_fold{fold}"
                fdir.mkdir(exist_ok=True)
                tr, va = train_val_split(tr_idx, y, groups, cfg["train"]["val_fraction"], seed)
                norm = FoldNormalizer(cfg).fit([graphs[i] for i in tr], seed)
                T = lambda gs: [norm.transform(g) for g in gs]  # noqa: E731
                train, val = T([graphs[i] for i in tr]), T([graphs[i] for i in va])
                test = T(external) if te_idx is None else T([graphs[i] for i in te_idx])
                ext_norm = T(extra) if extra else None
                model, hist = self.fit_fold(train, val, norm.dim, seed, ext_norm)
                df, nodes = self.predict(model, test, keep_nodes=True)
                if cfg["pseudotime"]["enabled"]:
                    pt = self.fit_pseudotime(model, train)
                    extra_cols = []
                    for cid, nd in nodes.items():
                        if len(nd["emb"]) == 0:
                            continue
                        nd["pseudotime"] = pt.transform(nd["emb"])
                        s = core_summary(nd["pseudotime"], nd["tumour_w"], cfg["pseudotime"]["bins"])
                        s["core_id"] = cid
                        extra_cols.append(s)
                    df = df.merge(pd.DataFrame(extra_cols), on="core_id", how="left")
                    with open(fdir / "pseudotime.pkl", "wb") as f:
                        pickle.dump(pt, f)
                df["fold"], df["rep"] = fold, rep
                all_preds.append(df)
                torch.save(model.state_dict(), fdir / "model.pt")
                with open(fdir / "normalizer.pkl", "wb") as f:
                    pickle.dump(norm, f)
                with open(fdir / "history.json", "w") as f:
                    json.dump(hist, f)
                if cfg["train"]["save_node_preds"]:
                    nd_dir = fdir / "node_preds"
                    nd_dir.mkdir(exist_ok=True)
                    for cid, nd in nodes.items():
                        nd.pop("emb", None)
                        np.savez_compressed(nd_dir / f"{cid}.npz", **nd)
                log.info("rep %d fold %d: %s", rep, fold, core_metrics(df.y.values, df.score.values))
        preds = pd.concat(all_preds, ignore_index=True)
        preds.to_csv(wd / "core_predictions.csv", index=False)
        for rep, d in preds.groupby("rep"):
            summary[f"rep{rep}"] = bootstrap_ci(d.y.values, d.score.values, cfg["train"]["bootstrap"], cfg["seed"])
        with open(wd / "metrics.json", "w") as f:
            json.dump(summary, f, indent=2)
        with open(wd / "config_used.json", "w") as f:
            json.dump(cfg, f, indent=2, default=str)
        return preds, summary
