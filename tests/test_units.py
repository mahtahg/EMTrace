import numpy as np
import pytest
import torch

from mesograph_fm.config import load_config
from mesograph_fm.extract import handcrafted as hc
from mesograph_fm.extract.foundation import DummyEncoder, mask_pooled_tokens
from mesograph_fm.extract.segmentation import read_cellvit_json
from mesograph_fm.extract.synthetic import labels_to_cellvit_json, make_core
from mesograph_fm.graph.build import radius_edges
from mesograph_fm.models.losses import pairwise_ranking_loss, proportion_loss
from mesograph_fm.models.mesograph import MesoGraphFM
from mesograph_fm.models.pseudotime import PrincipalTreePseudotime, core_summary


def original_ranking_loss(output, y, margin=0.8):
    """Loop version copied from the original MesoGraph _pair_train."""
    loss, c = 0, 0
    z = torch.zeros(1)
    for i in range(len(y) - 1):
        for j in range(i + 1, len(y)):
            if y[i] != y[j]:
                c += 1
                dz = output[i, :] - output[j, :]
                dy = torch.stack([torch.clamp(y[j] - y[i], -1, 1), torch.clamp(y[i] - y[j], -1, 1)])
                loss += torch.mean(torch.max(z, margin - dy * dz))
    return loss / c


def test_ranking_loss_matches_original():
    torch.manual_seed(0)
    out = torch.rand(9, 2)
    y = torch.tensor([0, 1, 2, 0, 2, 1, 1, 0, 2])
    assert torch.allclose(pairwise_ranking_loss(out, y, 0.8), original_ranking_loss(out, y, 0.8), atol=1e-6)


def test_ranking_loss_single_class_is_zero():
    out = torch.rand(4, 2, requires_grad=True)
    l = pairwise_ranking_loss(out, torch.zeros(4, dtype=torch.long))
    assert l.item() == 0.0
    l.backward()


def test_proportion_loss_bounds():
    cfg = load_config()["loss"]["proportion"]
    batch = torch.tensor([0] * 10 + [1] * 10 + [2] * 10)
    ei = torch.zeros((2, 0), dtype=torch.long)
    # perfect: E core all epithelioid, S all sarcomatoid, B half/half
    lg = torch.zeros(30, 2)
    lg[:10] = torch.tensor([10.0, -10.0])
    lg[10:15] = torch.tensor([10.0, -10.0])
    lg[15:20] = torch.tensor([-10.0, 10.0])
    lg[20:] = torch.tensor([-10.0, 10.0])
    y = torch.tensor([0, 1, 2])
    _, parts, f = proportion_loss(lg, batch, y, ei, cfg)
    assert parts["bound"] < 1e-6
    assert np.allclose(f.numpy(), [0, 0.5, 1], atol=1e-3)
    # violated: E core fully sarcomatoid
    _, parts2, _ = proportion_loss(-lg, batch, y, ei, cfg)
    assert parts2["bound"] > 0.1


def test_cellvit_json_roundtrip(tmp_path):
    img, lab, types = make_core("E", 256, 40, 0)
    labels_to_cellvit_json(lab, types, tmp_path / "c.json")
    lab2, probs, cents = read_cellvit_json(tmp_path / "c.json", lab.shape, 5)
    assert lab2.max() == len(probs) == len(np.unique(lab)) - 1
    # masks overlap strongly with the originals
    inter = ((lab > 0) & (lab2 > 0)).sum() / max((lab > 0).sum(), 1)
    assert inter > 0.85
    assert np.allclose(probs.sum(1), 1, atol=1e-5)
    assert (probs.argmax(1) == types[: len(probs)]).mean() > 0.95


def test_mask_pooling_is_exact_for_constant_tokens():
    class Const(DummyEncoder):
        def forward(self, x):
            B = x.shape[0]
            g = 224 // 14
            p = torch.arange(g * g, dtype=torch.float32).view(1, g, g, 1).repeat(B, 1, 1, self.dim)
            return torch.zeros(B, self.dim), p

    img = np.full((224, 224, 3), 200, np.uint8)
    lab = np.zeros((224, 224), np.int32)
    lab[0:14, 0:14] = 1  # token 0 exactly
    lab[14:28, 28:42] = 2  # token 16+2 = 18
    cfg = load_config()["extract"]["encoder"]
    out = mask_pooled_tokens(img, lab, 2, 0.5, Const(), dict(cfg, stride=224), "cpu")
    assert np.allclose(out["fm"][:, 0], [0, 18])


def test_handcrafted_shapes_and_spindle_signal():
    imgE, labE, tE = make_core("E", 256, 40, 1)
    imgS, labS, tS = make_core("S", 256, 40, 1)
    mE, names, _ = hc.morphology(labE, 0.5)
    mS, _, _ = hc.morphology(labS, 0.5)
    ar = names.index("aspect_ratio")
    assert np.nanmedian(mS[:, ar]) > 1.8 * np.nanmedian(mE[:, ar])
    ch = hc.stain_channels(imgE)
    i_, inames = hc.intensity(labE, ch, 0.5, 3.0)
    assert i_.shape == (labE.max(), len(inames)) == (labE.max(), 54)
    t_, tnames = hc.texture(labE[:128, :128], ch[:128, :128], 0.5, 50.0, 32)
    assert len(tnames) == 78 and t_.shape[1] == 78
    tp = np.eye(5)[tE]
    nb, nnames = hc.neighbourhood(np.random.rand(len(tE), 2) * 100, tp, [25, 50], ["a", "b", "c", "d", "e"])
    assert nb.shape == (len(tE), len(nnames))


def test_radius_edges_symmetric_and_bounded():
    pts = np.random.default_rng(0).random((200, 2)) * 200
    ei, d = radius_edges(pts, 20.0, 8)
    assert (d <= 20.0 + 1e-6).all()
    s = set(map(tuple, ei.T))
    assert all((b, a) in s for a, b in s)
    assert (ei[0] != ei[1]).all()


@pytest.mark.parametrize("conv,pool", [("edgeconv", "mean"), ("gatv2", "attention")])
def test_model_forward_backward(conv, pool):
    cfg = load_config()["model"]
    cfg = dict(cfg, conv=conv, pooling=pool)
    m = MesoGraphFM(12, cfg, edge_dim=4)
    x = torch.randn(30, 12)
    ei = torch.randint(0, 30, (2, 80))
    ea = torch.randn(80, 4)
    batch = torch.tensor([0] * 15 + [1] * 15)
    out = m(x, ei, batch, ea)
    assert out["core"].shape == (2, 2) and out["node"].shape == (30, 2)
    out["core"].sum().backward()


def test_pseudotime_orders_trajectory():
    rng = np.random.default_rng(0)
    t = rng.random(2000)
    emb = np.c_[t * 10, np.sin(t * 3), rng.normal(0, 0.05, (2000, 3))]
    pt = PrincipalTreePseudotime(pca_dim=3, n_centroids=20).fit(emb, q=t)
    p = pt.transform(emb)
    from scipy.stats import spearmanr

    assert spearmanr(p, t).statistic > 0.95
    s = core_summary(p[:100], None, 10)
    assert abs(sum(v for k, v in s.items() if k.startswith("pt_hist")) - 1) < 1e-6
