"""Interpretable per-nucleus features computed from CellViT masks.

Blocks
------
morph : shape (area, perimeter, axes, eccentricity, solidity, circularity,
        Hu moments, Fourier boundary descriptors, ...)
int   : H, E and optical-density intensity statistics in the nucleus and in a
        ring around it (cytoplasm proxy)
tex   : 13 Haralick features x 3 channels x {nucleus, 50 µm circle}
type  : CellViT cell-type probabilities
nbr   : neighbourhood composition / density at several radii
All lengths are in µm and areas in µm².
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.stats import kurtosis, skew
from skimage.color import rgb2hed
from skimage.measure import find_contours, regionprops

HARALICK_NAMES = [
    "asm", "contrast", "correlation", "variance", "idm", "sum_avg", "sum_var",
    "sum_entropy", "entropy", "diff_var", "diff_entropy", "imc1", "imc2",
]
CHANNELS = ["H", "E", "OD"]
STATS = ["mean", "std", "min", "max", "median", "p10", "p90", "skew", "kurt"]


def stain_channels(image: np.ndarray) -> np.ndarray:
    """(H,W,3) uint8 -> (H,W,3) float32 [hematoxylin, eosin, OD sum]."""
    rgb = image.astype(np.float32)
    hed = rgb2hed(np.clip(rgb, 1, 255) / 255.0)
    od = -np.log(np.clip(rgb, 1, 255) / 255.0).sum(-1)
    return np.stack([hed[..., 0], hed[..., 1], od], -1).astype(np.float32)


def _fourier(contour: np.ndarray, k: int) -> np.ndarray:
    if len(contour) < 4:
        return np.full(k, np.nan)
    z = contour[:, 1] + 1j * contour[:, 0]
    # resample uniformly by arc length
    d = np.r_[0, np.cumsum(np.abs(np.diff(z)))]
    if d[-1] <= 0:
        return np.full(k, np.nan)
    t = np.linspace(0, d[-1], 64, endpoint=False)
    zr = np.interp(t, d, z.real) + 1j * np.interp(t, d, z.imag)
    F = np.abs(np.fft.fft(zr - zr.mean()))
    base = F[1] if F[1] > 0 else 1.0
    out = F[2:2 + k] / base  # scale/rotation/translation invariant
    return np.pad(out, (0, k - len(out)), constant_values=0)


def morphology(labels: np.ndarray, mpp: float, n_harm: int = 10):
    names = [
        "area", "perimeter", "major_axis", "minor_axis", "aspect_ratio", "eccentricity",
        "solidity", "extent", "circularity", "convexity", "equiv_diameter", "feret_max",
        "orient_sin2", "orient_cos2", "inertia_ratio",
    ] + [f"hu{i}" for i in range(7)] + [f"fourier{i}" for i in range(n_harm)]
    props = regionprops(labels)
    out = np.full((len(props), len(names)), np.nan, dtype=np.float32)
    for i, p in enumerate(props):
        A = p.area * mpp ** 2
        P = max(p.perimeter, 1e-6) * mpp
        ma, mi = p.axis_major_length * mpp, p.axis_minor_length * mpp
        try:
            convP = regionprops(p.image_convex.astype(np.uint8))[0].perimeter * mpp
        except Exception:
            convP = np.nan
        ev = p.inertia_tensor_eigvals
        hu = p.moments_hu
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
        cs = find_contours(np.pad(p.image.astype(float), 1), 0.5)
        four = _fourier(max(cs, key=len), n_harm) if cs else np.full(n_harm, np.nan)
        try:
            feret = p.feret_diameter_max * mpp
        except Exception:
            feret = np.nan
        out[i] = [
            A, P, ma, mi, ma / max(mi, 1e-6), p.eccentricity, p.solidity, p.extent,
            4 * np.pi * A / P ** 2, convP / P, p.equivalent_diameter_area * mpp, feret,
            np.sin(2 * p.orientation), np.cos(2 * p.orientation), ev[1] / max(ev[0], 1e-9),
            *hu, *four,
        ]
    cents = np.array([p.centroid for p in props]).reshape(-1, 2)  # (row, col) px
    return out, names, cents


def _stats(v: np.ndarray) -> list[float]:
    if v.size < 2:
        return [np.nan] * len(STATS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q = np.percentile(v, [10, 50, 90])
        return [v.mean(), v.std(), v.min(), v.max(), q[1], q[0], q[2], skew(v), kurtosis(v)]


def intensity(labels: np.ndarray, chans: np.ndarray, mpp: float, ring_um: float):
    names = [f"{reg}_{ch}_{s}" for reg in ("nuc", "ring") for ch in CHANNELS for s in STATS]
    r = max(int(round(ring_um / mpp)), 1)
    props = regionprops(labels)
    out = np.full((len(props), len(names)), np.nan, dtype=np.float32)
    struct = ndi.generate_binary_structure(2, 1)
    for i, p in enumerate(props):
        r0, c0, r1, c1 = p.bbox
        R0, C0 = max(r0 - r, 0), max(c0 - r, 0)
        R1, C1 = min(r1 + r, labels.shape[0]), min(c1 + r, labels.shape[1])
        lab = labels[R0:R1, C0:C1]
        nuc = lab == p.label
        ring = ndi.binary_dilation(nuc, struct, iterations=r) & (lab == 0)
        ch = chans[R0:R1, C0:C1]
        row = []
        for m in (nuc, ring):
            for k in range(3):
                row += _stats(ch[..., k][m])
        out[i] = row
    return out, names


def _quantise(x: np.ndarray, lo: float, hi: float, levels: int) -> np.ndarray:
    q = np.floor((x - lo) / max(hi - lo, 1e-8) * (levels - 1)).astype(np.int64)
    return np.clip(q, 0, levels - 1) + 1  # 1..levels, 0 reserved for "ignore"


def texture(labels: np.ndarray, chans: np.ndarray, mpp: float, circle_um: float, levels: int):
    import mahotas

    names = [f"{reg}_{ch}_{h}" for reg in ("nuc", "circ") for ch in CHANNELS for h in HARALICK_NAMES]
    props = regionprops(labels)
    out = np.full((len(props), len(names)), np.nan, dtype=np.float32)
    lo = np.percentile(chans.reshape(-1, 3), 1, axis=0)
    hi = np.percentile(chans.reshape(-1, 3), 99, axis=0)
    rad = max(int(round(circle_um / 2 / mpp)), 2)
    yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1]
    disk = (yy ** 2 + xx ** 2) <= rad ** 2
    H, W = labels.shape
    for i, p in enumerate(props):
        row = []
        # nucleus region
        r0, c0, r1, c1 = p.bbox
        m_nuc = labels[r0:r1, c0:c1] == p.label
        # circle region centred on centroid
        cy, cx = int(round(p.centroid[0])), int(round(p.centroid[1]))
        y0, y1, x0, x1 = cy - rad, cy + rad + 1, cx - rad, cx + rad + 1
        dy0, dx0 = max(-y0, 0), max(-x0, 0)
        dy1, dx1 = max(y1 - H, 0), max(x1 - W, 0)
        m_circ = disk[dy0:disk.shape[0] - dy1, dx0:disk.shape[1] - dx1]
        regions = [
            (chans[r0:r1, c0:c1], m_nuc),
            (chans[max(y0, 0):min(y1, H), max(x0, 0):min(x1, W)], m_circ),
        ]
        for ch_img, m in regions:
            for k in range(3):
                q = _quantise(ch_img[..., k], lo[k], hi[k], levels) * m
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        h = mahotas.features.haralick(q, ignore_zeros=True, return_mean=True)
                except (ValueError, ZeroDivisionError):
                    h = np.full(13, np.nan)
                row += list(h[:13])
        out[i] = row
    return out, names


def neighbourhood(cents_um: np.ndarray, type_probs: np.ndarray, radii_um, type_names):
    names = []
    n, T = type_probs.shape
    cols = []
    tree = cKDTree(cents_um) if n else None
    for r in radii_um:
        names += [f"r{int(r)}_count", f"r{int(r)}_density_per_mm2"] + [f"r{int(r)}_frac_{t}" for t in type_names]
        if n == 0:
            continue
        nb = tree.query_ball_point(cents_um, r)
        cnt = np.array([len(x) - 1 for x in nb], dtype=np.float32)
        frac = np.stack([type_probs[[j for j in x if j != i]].sum(0) if len(x) > 1 else np.zeros(T)
                         for i, x in enumerate(nb)]) / np.maximum(cnt[:, None], 1)
        cols += [cnt[:, None], (cnt / (np.pi * r ** 2) * 1e6)[:, None], frac]
    names += ["nn1_dist", "nn5_mean_dist", "nn5_std_dist"] + [f"dist_nearest_{t}" for t in type_names]
    if n == 0:
        return np.zeros((0, len(names)), dtype=np.float32), names
    k = min(6, n)
    d, _ = tree.query(cents_um, k=k)
    d = d.reshape(n, k)[:, 1:] if k > 1 else np.full((n, 1), np.nan)
    cols.append(np.stack([d[:, 0], d[:, :5].mean(1), d[:, :5].std(1)], 1))
    hard = type_probs.argmax(1)
    dn = np.full((n, T), np.nan, dtype=np.float32)
    for t in range(T):
        idx = np.where(hard == t)[0]
        if len(idx) == 0:
            continue
        tr = cKDTree(cents_um[idx])
        kk = min(2, len(idx))
        dd, jj = tr.query(cents_um, k=kk)
        dd, jj = dd.reshape(n, kk), jj.reshape(n, kk)
        self_hit = idx[jj[:, 0]] == np.arange(n)
        dn[:, t] = np.where(self_hit, dd[:, -1] if kk > 1 else np.nan, dd[:, 0])
    cols.append(dn)
    return np.concatenate(cols, 1).astype(np.float32), names
