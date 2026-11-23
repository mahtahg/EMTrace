"""Novelty F: pathologist-concept scores per nucleus.

A vision-language pathology model (CONCH, Lu et al. 2024) scores a small
crop around every nucleus against text prompts for concepts pathologists use
to subtype mesothelioma (spindle shape, pleomorphism, nucleoli, mitoses,
necrosis, stroma type, ...). Each concept's prompts are embedded and averaged;
the per-cell score is the cosine similarity between the crop embedding and
the concept embedding. These become the ``concept`` feature block, which is
human-readable and is reported separately by the explainer.

CONCH is gated on Hugging Face (MahmoodLab/conch) and must be installed from
https://github.com/mahmoodlab/CONCH; set ``HF_TOKEN``.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from skimage.transform import resize


class ConceptScorer:
    def __init__(self, prompts: dict[str, list[str]], device):
        self.names = list(prompts)
        self.prompts = prompts
        self.device = device

    def encode_images(self, crops: np.ndarray) -> torch.Tensor:  # (B,S,S,3) uint8 -> (B,D) normed
        raise NotImplementedError

    def text_matrix(self) -> torch.Tensor:  # (C,D) normed
        raise NotImplementedError

    input_size = 448

    @torch.no_grad()
    def score(self, image: np.ndarray, cents_rc: np.ndarray, mpp: float, crop_um: float, batch_size: int):
        if len(cents_rc) == 0:
            return np.zeros((0, len(self.names)), dtype=np.float32)
        T = self.text_matrix()
        half = max(int(round(crop_um / 2 / mpp)), 4)
        padded = np.pad(image, ((half, half), (half, half), (0, 0)), mode="reflect")
        out = []
        for b0 in range(0, len(cents_rc), batch_size):
            cs = cents_rc[b0:b0 + batch_size]
            crops = []
            for r, c in cs:
                r, c = int(round(r)) + half, int(round(c)) + half
                crop = padded[r - half:r + half, c - half:c + half]
                crops.append(resize(crop, (self.input_size, self.input_size), order=1,
                                    preserve_range=True).astype(np.uint8))
            emb = self.encode_images(np.stack(crops))
            out.append((emb @ T.T).float().cpu().numpy())
        return np.concatenate(out).astype(np.float32)


class ConchScorer(ConceptScorer):
    def __init__(self, prompts, device, hf_token_env="HF_TOKEN"):
        super().__init__(prompts, device)
        from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize

        self.model, self.preprocess = create_model_from_pretrained(
            "conch_ViT-B-16", "hf_hub:MahmoodLab/conch", hf_auth_token=os.environ.get(hf_token_env)
        )
        self.model = self.model.to(device).eval()
        self._tok, self._tokenize = get_tokenizer(), tokenize
        self._T = None

    @torch.no_grad()
    def text_matrix(self):
        if self._T is None:
            rows = []
            for n in self.names:
                ids = self._tokenize(texts=self.prompts[n], tokenizer=self._tok).to(self.device)
                e = self.model.encode_text(ids)
                e = torch.nn.functional.normalize(e, dim=-1).mean(0)
                rows.append(torch.nn.functional.normalize(e, dim=-1))
            self._T = torch.stack(rows)
        return self._T

    @torch.no_grad()
    def encode_images(self, crops):
        from PIL import Image

        x = torch.stack([self.preprocess(Image.fromarray(c)) for c in crops]).to(self.device)
        e = self.model.encode_image(x, proj_contrast=True, normalize=True)
        return e


class DummyScorer(ConceptScorer):
    """Deterministic stand-in: concept scores from simple colour statistics."""

    input_size = 32

    def __init__(self, prompts, device):
        super().__init__(prompts, device)
        g = torch.Generator().manual_seed(1)
        self.W = torch.randn(len(self.names), 6, generator=g)

    def text_matrix(self):
        return torch.nn.functional.normalize(self.W, dim=-1)

    def encode_images(self, crops):
        x = torch.from_numpy(crops.astype(np.float32) / 255.0)
        f = torch.cat([x.mean((1, 2)), x.std((1, 2))], 1)
        return torch.nn.functional.normalize(f, dim=-1)


def build_scorer(cfg: dict, device) -> ConceptScorer | None:
    c = cfg["extract"]["concepts"]
    if not c["enabled"]:
        return None
    if c["backend"] == "conch":
        return ConchScorer(c["prompts"], device, c["hf_token_env"])
    if c["backend"] == "dummy":
        return DummyScorer(c["prompts"], device)
    raise ValueError(f"unknown concept backend {c['backend']}")
