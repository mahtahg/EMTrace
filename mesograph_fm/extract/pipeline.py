"""Run per-core feature extraction and write one ``<core_id>.npz`` per core.

Each npz holds ``coords_um`` (N,2 as x,y), ``type_probs`` (N,T), one array per
feature block (``blk_<name>``) and ``names_<name>`` with column names.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.measure import regionprops

from ..config import resolve_device
from . import handcrafted as hc
from .concepts import build_scorer
from .foundation import build_encoder, mask_pooled_tokens
from .segmentation import segment_core

log = logging.getLogger(__name__)


def load_image(path: str) -> np.ndarray:
    if str(path).lower().endswith((".tif", ".tiff")):
        import tifffile

        img = tifffile.imread(path)
    else:
        img = np.asarray(Image.open(path).convert("RGB"))
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)
    return img[..., :3].astype(np.uint8)


def extract_core(row: dict, cfg: dict, encoder=None, scorer=None, device="cpu") -> dict:
    ex = cfg["extract"]
    mpp = float(row["mpp"])
    image = load_image(row["image"])
    seg = segment_core(row, image, cfg)
    labels, tp = seg.labels, seg.type_probs
    n = seg.n
    props = regionprops(labels)
    assert len(props) == n, "label map and type table out of sync"
    cents_rc = np.array([p.centroid for p in props]).reshape(-1, 2)
    coords_um = cents_rc[:, ::-1] * mpp  # x, y in µm
    out: dict[str, np.ndarray] = {"coords_um": coords_um.astype(np.float32), "type_probs": tp}
    names: dict[str, list[str]] = {}

    hh = ex["handcrafted"]
    chans = hc.stain_channels(image) if (hh["intensity"] or hh["texture"]) else None
    if hh["morphology"]:
        out["blk_morph"], names["morph"], _ = hc.morphology(labels, mpp, hh["fourier_harmonics"])
    if hh["intensity"]:
        out["blk_int"], names["int"] = hc.intensity(labels, chans, mpp, hh["ring_um"])
    if hh["texture"]:
        out["blk_tex"], names["tex"] = hc.texture(labels, chans, mpp, hh["texture_circle_um"], hh["texture_levels"])
    out["blk_type"], names["type"] = tp.astype(np.float32), [f"p_{t}" for t in ex["cell_types"]]
    if hh["neighbourhood"]:
        out["blk_nbr"], names["nbr"] = hc.neighbourhood(coords_um, tp, hh["neighbourhood_radii_um"], ex["cell_types"])
    if seg.cvit_emb is not None:
        out["blk_cvit"] = seg.cvit_emb
        names["cvit"] = [f"cvit{i}" for i in range(seg.cvit_emb.shape[1])]
    if encoder is not None and n:
        fm = mask_pooled_tokens(image, labels, n, mpp, encoder, ex["encoder"], device)
        for k, v in fm.items():
            out[f"blk_{k}"] = v
            names[k] = [f"{k}{i}" for i in range(v.shape[1])]
    if scorer is not None:
        out["blk_concept"] = scorer.score(image, cents_rc, mpp, ex["concepts"]["crop_um"], ex["concepts"]["batch_size"])
        names["concept"] = [f"concept_{c}" for c in scorer.names]
    for k, v in names.items():
        out[f"names_{k}"] = np.array(json.dumps(v))
    return out


def extract_dataset(cfg: dict, csv: str | None = None, overwrite: bool = False):
    df = pd.read_csv(csv or cfg["paths"]["dataset_csv"]).fillna("")
    outdir = Path(cfg["paths"]["features_dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg)
    encoder = build_encoder(cfg)
    scorer = build_scorer(cfg, device)
    for row in df.to_dict("records"):
        dst = outdir / f"{row['core_id']}.npz"
        if dst.exists() and not overwrite:
            continue
        res = extract_core(row, cfg, encoder, scorer, device)
        np.savez_compressed(dst, **res)
        log.info("extracted %s: %d cells", row["core_id"], len(res["coords_um"]))
    return outdir
