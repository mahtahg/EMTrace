"""Full pipeline on synthetic cores: extract -> graphs -> pretrain G -> train (A)
-> pseudotime C -> explain, with concepts F on. Uses dummy encoder/scorer
so no gated weights are needed."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from mesograph_fm.config import load_config
from mesograph_fm.extract.pipeline import extract_dataset
from mesograph_fm.extract.synthetic import make_dataset
from mesograph_fm.graph.build import build_dataset, load_graphs
from mesograph_fm.graph.normalize import FoldNormalizer
from mesograph_fm.train.engine import Trainer


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    wd = tmp_path_factory.mktemp("run")
    make_dataset(wd / "data", n_per_class=6, size=320, n_cells=70)
    cfg = load_config(None, {
        "paths": {"dataset_csv": str(wd / "data" / "cores.csv"), "work_dir": str(wd)},
        "extract": {"encoder": {"name": "dummy", "batch_size": 8}, "concepts": {"backend": "dummy"}},
        "features": {"pca_components": 8},
        "train": {"epochs": 15, "split": "kfold", "n_folds": 3, "bootstrap": 50, "log_every": 100},
        "pretrain": {"epochs": 3},
        "pseudotime": {"n_centroids": 10},
    })
    extract_dataset(cfg)
    build_dataset(cfg)
    return cfg, wd


def test_features_written(run):
    cfg, wd = run
    z = np.load(wd / "features" / "core000.npz")
    for b in ["morph", "int", "tex", "type", "nbr", "fm", "ctx", "concept"]:
        assert f"blk_{b}" in z.files, b
        assert len(z[f"blk_{b}"]) == len(z["coords_um"])
        assert len(json.loads(str(z[f"names_{b}"]))) == z[f"blk_{b}"].shape[1]
    n_feat = sum(z[f"blk_{b}"].shape[1] for b in ["morph", "int", "tex", "type", "nbr", "concept"])
    assert n_feat >= 200


def test_normalizer_uses_train_only(run):
    cfg, wd = run
    gs = load_graphs(wd / "graphs")
    a = FoldNormalizer(cfg).fit(gs[:6])
    b = FoldNormalizer(cfg).fit(gs[:6] + gs[6:])
    assert not np.allclose(a.stats["morph"]["med"], b.stats["morph"]["med"])
    x = a.transform(gs[10]).x
    assert x.shape[1] == a.dim and torch.isfinite(x).all()


def test_train_full_pipeline(run):
    cfg, wd = run
    preds, summary = Trainer(cfg).run()
    assert len(preds) == 18 and set(preds.core_id) == {f"core{i:03d}" for i in range(18)}
    for c in ["score", "sarcomatoid_pct", "pt_mean", "pt_hist0"]:
        assert c in preds and preds[c].notna().all(), c
    assert preds.sarcomatoid_pct.between(0, 100).all()
    assert (wd / "metrics.json").exists() and (wd / "rep0_fold0" / "model.pt").exists()
    hist = json.load(open(wd / "rep0_fold0" / "history.json"))
    assert len(hist["pretrain"]) == 3  # G on
    # synthetic task is easy: spindle cores should score higher than round ones
    m = summary["rep0"]["auc_E_vs_S"]["value"]
    assert m > 0.7, m


def test_toggles_off(run, tmp_path):
    cfg, wd = run
    import copy

    c2 = copy.deepcopy(cfg)
    c2["paths"]["work_dir"] = str(tmp_path)
    c2["pretrain"]["enabled"] = False
    c2["loss"]["proportion"]["enabled"] = False
    c2["pseudotime"]["enabled"] = False
    c2["model"]["conv"] = "gatv2"
    c2["train"]["epochs"] = 3
    preds, _ = Trainer(c2).run(graphs=load_graphs(wd / "graphs"))
    assert "pt_mean" not in preds
    hist = json.load(open(tmp_path / "rep0_fold0" / "history.json"))
    assert hist["pretrain"] == []


def test_explain(run):
    import pickle

    from mesograph_fm.explain.attribution import explain_cores
    from mesograph_fm.models.mesograph import MesoGraphFM

    cfg, wd = run
    fdir = wd / "rep0_fold0"
    norm = pickle.load(open(fdir / "normalizer.pkl", "rb"))
    gs = [norm.transform(g) for g in load_graphs(wd / "graphs")[:3]]
    m = MesoGraphFM(norm.dim, cfg["model"], gs[0].edge_attr.shape[1])
    m.load_state_dict(torch.load(fdir / "model.pt"))
    feats, blocks = explain_cores(m, gs, norm, steps=4)
    assert feats.shape == (3, norm.dim)
    assert set(blocks.columns) == set(norm.slices)


def test_tiny_core_does_not_crash(run, tmp_path):
    """A core with a single nucleus must flow through normalise/predict/pseudotime."""
    from mesograph_fm.extract.synthetic import make_core
    from mesograph_fm.extract.pipeline import extract_core
    from mesograph_fm.graph.build import build_graph
    from PIL import Image

    cfg, wd = run
    img, lab, types = make_core("E", 128, 1, 3)
    Image.fromarray(img).save(tmp_path / "t.png")
    np.save(tmp_path / "t_mask.npy", lab)
    import copy
    c = copy.deepcopy(cfg)
    c["extract"]["segmenter"] = "label_map"
    res = extract_core({"image": str(tmp_path / "t.png"), "mpp": 0.5, "mask": str(tmp_path / "t_mask.npy")}, c,
                       __import__("mesograph_fm.extract.foundation", fromlist=["x"]).DummyEncoder(),
                       __import__("mesograph_fm.extract.concepts", fromlist=["x"]).DummyScorer(c["extract"]["concepts"]["prompts"], "cpu"))
    np.savez(tmp_path / "t.npz", **res)
    g = build_graph(tmp_path / "t.npz", "E", "tiny", "g", c)
    gs = load_graphs(wd / "graphs")
    norm = FoldNormalizer(c).fit(gs)
    tr = Trainer(c)
    from mesograph_fm.models.mesograph import MesoGraphFM
    m = MesoGraphFM(norm.dim, c["model"], 4)
    df, nodes = tr.predict(m, [norm.transform(g)], keep_nodes=True)
    assert len(df) == 1 and np.isfinite(df.score).all()
