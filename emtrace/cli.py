"""Command-line interface: ``emtrace <command> --config cfg.yaml``.

Commands
--------
extract   segment-file reading + per-nucleus features -> <work_dir>/features
graphs    build cell graphs                          -> <work_dir>/graphs
train     cross-validated training (+ pretraining, proportion loss, pseudotime)
explain   integrated gradients + GNNExplainer for a trained fold
demo      make a synthetic dataset and run the full pipeline (no weights needed)
"""
from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import pandas as pd
import torch

from .config import load_config, resolve_device


def _cfg(a):
    ov = {"paths": {"work_dir": a.work_dir}} if getattr(a, "work_dir", None) else None
    return load_config(a.config, ov)


def cmd_extract(a):
    from .extract.pipeline import extract_dataset

    cfg = _cfg(a)
    extract_dataset(cfg, overwrite=a.overwrite)
    if cfg["paths"]["external_csv"]:
        ext = dict(cfg, paths=dict(cfg["paths"], features_dir=str(Path(cfg["paths"]["features_dir"]) / "external")))
        extract_dataset(ext, cfg["paths"]["external_csv"], a.overwrite)


def cmd_graphs(a):
    from .graph.build import build_dataset

    cfg = _cfg(a)
    build_dataset(cfg)
    if cfg["paths"]["external_csv"]:
        ext = dict(cfg, paths=dict(cfg["paths"], features_dir=str(Path(cfg["paths"]["features_dir"]) / "external")))
        build_dataset(ext, cfg["paths"]["external_csv"], str(Path(cfg["paths"]["graphs_dir"]) / "external"))


def cmd_train(a):
    from .graph.build import load_graphs
    from .train.engine import Trainer

    cfg = _cfg(a)
    ext = None
    if cfg["train"]["split"] == "external":
        ext = load_graphs(Path(cfg["paths"]["graphs_dir"]) / "external")
    preds, summary = Trainer(cfg).run(external=ext)
    print(pd.DataFrame({k: {m: v["value"] for m, v in s.items()} for k, s in summary.items()}).round(3))


def cmd_explain(a):
    from .explain.attribution import explain_cores
    from .graph.build import load_graphs
    from .models.emtrace_net import EMTraceNet

    cfg = _cfg(a)
    fdir = Path(cfg["paths"]["work_dir"]) / a.fold
    with open(fdir / "normalizer.pkl", "rb") as f:
        norm = pickle.load(f)
    raw = load_graphs(cfg["paths"]["graphs_dir"])
    graphs = [norm.transform(g) for g in raw]
    ed = graphs[0].edge_attr.shape[1] if getattr(graphs[0], "edge_attr", None) is not None else None
    model = EMTraceNet(norm.dim, cfg["model"], ed)
    dev = resolve_device(cfg)
    model.load_state_dict(torch.load(fdir / "model.pt", map_location=dev))
    model.to(dev)
    ec = cfg["explain"]
    methods = a.method.split(",") if a.method else ec["methods"]
    cell_dir = None
    if ec["save_cell_maps"]:
        cell_dir = fdir / "explanations"
        cell_dir.mkdir(exist_ok=True)
    res = explain_cores(model, graphs, norm, dev, ec["ig_steps"], methods, ec["gnnexplainer_epochs"], cell_dir)
    for m, (feats, blocks) in res.items():
        feats.to_csv(fdir / f"attributions_{m}_features.csv")
        blocks.to_csv(fdir / f"attributions_{m}_blocks.csv")
        print(f"\n[{m}] mean |attribution| per block")
        print(blocks.abs().mean().sort_values(ascending=False))

def cmd_demo(a):
    from .extract.pipeline import extract_dataset
    from .extract.synthetic import make_dataset
    from .graph.build import build_dataset
    from .train.engine import Trainer

    wd = Path(a.work_dir or "runs/demo")
    make_dataset(wd / "data", n_per_class=a.n_per_class)
    cfg = load_config(a.config, {
        "paths": {"dataset_csv": str(wd / "data" / "cores.csv"), "work_dir": str(wd)},
        "extract": {"encoder": {"name": "dummy"}, "concepts": {"backend": "dummy"}},
        "train": {"epochs": a.epochs, "split": "kfold", "n_folds": 3, "bootstrap": 100},
        "pretrain": {"epochs": max(a.epochs // 4, 2)},
    })
    extract_dataset(cfg)
    build_dataset(cfg)
    preds, summary = Trainer(cfg).run()
    print(preds.head())
    print(summary)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser("emtrace")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in [("extract", cmd_extract), ("graphs", cmd_graphs), ("train", cmd_train),
                     ("explain", cmd_explain), ("demo", cmd_demo)]:
        s = sub.add_parser(name)
        s.add_argument("--config", default=None)
        s.add_argument("--work-dir", dest="work_dir", default=None)
        s.set_defaults(fn=fn)
        if name == "extract":
            s.add_argument("--overwrite", action="store_true")
        if name == "explain":
            s.add_argument("--fold", default="rep0_fold0")
            s.add_argument("--method", default=None, help="ig, gnnexplainer, or ig,gnnexplainer (default: config)")
        if name == "demo":
            s.add_argument("--epochs", type=int, default=20)
            s.add_argument("--n-per-class", dest="n_per_class", type=int, default=6)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
