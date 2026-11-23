"""Foundation-model (H-optimus-1) per-nucleus features.

The core is resampled to the encoder's resolution (0.5 µm/px for H-optimus-1)
and tiled with overlap. Each tile is encoded once; every nucleus receives the
average of the patch tokens its pixels fall in, weighted by pixel overlap and
accumulated across overlapping tiles ("mask-pooled token features", block
``fm``). Optionally each nucleus also receives the CLS token of the tiles it
lies in (block ``ctx``). This is ~100x cheaper than encoding one crop per cell
and keeps features nucleus-specific at the encoder's 7 µm token resolution.

H-optimus-1 (Bioptimus) is CC BY-NC-ND 4.0 and gated on Hugging Face: users
must accept its terms and set ``HF_TOKEN``. No weights are bundled here.
"""
from __future__ import annotations

import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from skimage.transform import resize

HOPT_MEAN = (0.707223, 0.578729, 0.703617)
HOPT_STD = (0.211883, 0.230117, 0.177517)


class TokenEncoder(nn.Module):
    """Interface: forward(images[B,3,S,S] normalised) -> (cls[B,D], patch[B,g,g,D])."""

    patch_size: int
    dim: int
    mean = HOPT_MEAN
    std = HOPT_STD


class HOptimus1(TokenEncoder):
    patch_size = 14
    dim = 1536

    def __init__(self, hf_token_env: str = "HF_TOKEN"):
        super().__init__()
        import timm

        tok = os.environ.get(hf_token_env)
        if tok:
            try:
                from huggingface_hub import login

                login(token=tok, add_to_git_credential=False)
            except Exception:  # pragma: no cover
                pass
        self.model = timm.create_model(
            "hf-hub:bioptimus/H-optimus-1", pretrained=True, init_values=1e-5, dynamic_img_size=False
        )
        self.model.eval()
        self.n_prefix = getattr(self.model, "num_prefix_tokens", 1)

    @torch.no_grad()
    def forward(self, x):
        tokens = self.model.forward_features(x)  # (B, prefix + g*g, D)
        cls = tokens[:, 0]
        patch = tokens[:, self.n_prefix:]
        g = int(round(patch.shape[1] ** 0.5))
        return cls, patch.reshape(x.shape[0], g, g, -1)


class DummyEncoder(TokenEncoder):
    """Tiny random patch embedder with the same interface (tests only)."""

    patch_size = 14
    dim = 32

    def __init__(self, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.proj = nn.Conv2d(3, self.dim, self.patch_size, self.patch_size)
        with torch.no_grad():
            self.proj.weight.copy_(torch.randn(self.proj.weight.shape, generator=g) * 0.05)
            self.proj.bias.zero_()

    @torch.no_grad()
    def forward(self, x):
        p = self.proj(x).permute(0, 2, 3, 1)  # B,g,g,D
        return p.mean((1, 2)), p


def build_encoder(cfg: dict) -> TokenEncoder | None:
    name = cfg["extract"]["encoder"]["name"]
    if name == "none":
        return None
    if name == "hoptimus1":
        return HOptimus1(cfg["extract"]["encoder"]["hf_token_env"])
    if name == "dummy":
        return DummyEncoder()
    raise ValueError(f"unknown encoder {name}")


def _tile_starts(n: int, tile: int, stride: int):
    if n <= tile:
        return [0]
    s = list(range(0, n - tile + 1, stride))
    if s[-1] != n - tile:
        s.append(n - tile)
    return s


def mask_pooled_tokens(image: np.ndarray, labels: np.ndarray, n_cells: int, mpp: float,
                       encoder: TokenEncoder, enc_cfg: dict, device) -> dict[str, np.ndarray]:
    """Return {"fm": (N,D)} and optionally {"ctx": (N,D)} per-nucleus features."""
    scale = mpp / enc_cfg["mpp"]  # >1 => upsample
    H, W = labels.shape
    h2, w2 = max(int(round(H * scale)), 1), max(int(round(W * scale)), 1)
    img = resize(image, (h2, w2), order=1, preserve_range=True, anti_aliasing=scale < 1).astype(np.float32) / 255.0
    lab = resize(labels, (h2, w2), order=0, preserve_range=True, anti_aliasing=False).astype(np.int64)
    T, st, P = enc_cfg["tile_size"], enc_cfg["stride"], encoder.patch_size
    pad_h, pad_w = max(T - h2, 0), max(T - w2, 0)
    if pad_h or pad_w:
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=1.0)
        lab = np.pad(lab, ((0, pad_h), (0, pad_w)))
    g = T // P
    mean = torch.tensor(encoder.mean).view(1, 3, 1, 1)
    std = torch.tensor(encoder.std).view(1, 3, 1, 1)
    D = encoder.dim
    num_fm = np.zeros((n_cells, D), dtype=np.float64)
    num_ctx = np.zeros((n_cells, D), dtype=np.float64)
    den = np.zeros(n_cells, dtype=np.float64)

    coords = [(y, x) for y in _tile_starts(img.shape[0], T, st) for x in _tile_starts(img.shape[1], T, st)]
    tok_idx = (np.arange(T)[:, None] // P) * g + (np.arange(T)[None, :] // P)  # (T,T) token id per pixel
    tok_idx = np.minimum(tok_idx, g * g - 1)
    encoder = encoder.to(device)
    bs = enc_cfg["batch_size"]
    for b0 in range(0, len(coords), bs):
        batch = coords[b0:b0 + bs]
        labs = [lab[y:y + T, x:x + T] for y, x in batch]
        if all((l_ == 0).all() for l_ in labs):
            continue
        x = torch.from_numpy(np.stack([img[y:y + T, x_:x_ + T] for y, x_ in batch])).permute(0, 3, 1, 2)
        x = ((x - mean) / std).to(device)
        cls, patch = encoder(x)
        cls = cls.float().cpu().numpy()
        patch = patch.float().cpu().numpy().reshape(len(batch), g * g, D)
        for i, l_ in enumerate(labs):
            m = l_ > 0
            if not m.any():
                continue
            cells = l_[m] - 1
            toks = tok_idx[m]
            A = sp.coo_matrix((np.ones(len(cells)), (cells, toks)), shape=(n_cells, g * g)).tocsr()
            cnt = np.asarray(A.sum(1)).ravel()
            num_fm += A @ patch[i]
            num_ctx += cnt[:, None] * cls[i][None]
            den += cnt
    den_ = np.maximum(den, 1e-8)[:, None]
    out = {"fm": (num_fm / den_).astype(np.float32)}
    if enc_cfg.get("use_tile_context", True):
        out["ctx"] = (num_ctx / den_).astype(np.float32)
    miss = den == 0
    for v in out.values():
        v[miss] = np.nan
    return out
