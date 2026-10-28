"""Place features: frozen DINOv2 ViT-S/14, 384-d, L2-normalised. Full 640x480 frame -> 224x168 (no centre crop),
ImageNet mean/std, CLS token (or mean of patch tokens via cfg.pooling). Model loaded once; fp16 + no_grad."""
from __future__ import annotations

from typing import List, Sequence

import cv2
import numpy as np

from .config import FeatureCfg


def preprocess(bgr: np.ndarray, cfg: FeatureCfg) -> np.ndarray:
    """BGR uint8 HxWx3 -> float32 CxHxW normalised. Pure numpy/cv2 (tested without torch)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    r = cv2.resize(rgb, (cfg.input_w, cfg.input_h), interpolation=cv2.INTER_AREA)
    x = (r.astype(np.float32) / 255.0 - np.array(cfg.mean, np.float32)) / np.array(cfg.std, np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def pool_tokens(tokens: np.ndarray, pooling: str) -> np.ndarray:
    """tokens (B, 1+N, D): index 0 = CLS. Returns (B, D)."""
    if pooling == "cls":
        return tokens[:, 0]
    if pooling == "mean":
        return tokens[:, 1:].mean(axis=1)
    raise ValueError(f"pooling must be 'cls' or 'mean', got {pooling!r}")


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


class Embedder:
    def __init__(self, cfg: FeatureCfg):
        import torch
        from transformers import AutoModel
        self.cfg, self.torch = cfg, torch
        self.device = cfg.device if cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(cfg.model_id)
        self.dtype = torch.float16 if (cfg.fp16 and self.device == "cuda") else torch.float32
        self.model = self.model.to(self.device, dtype=self.dtype).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.n_tokens = 1 + (cfg.input_h // 14) * (cfg.input_w // 14)

    def embed(self, bgr_list: Sequence[np.ndarray]) -> np.ndarray:
        torch, out = self.torch, []
        with torch.no_grad():
            for i in range(0, len(bgr_list), self.cfg.batch_size):
                x = np.stack([preprocess(b, self.cfg) for b in bgr_list[i:i + self.cfg.batch_size]])
                h = self.model(pixel_values=torch.from_numpy(x).to(self.device, dtype=self.dtype)).last_hidden_state
                if h.shape[1] != self.n_tokens:
                    raise RuntimeError(f"unexpected token count {h.shape[1]} != {self.n_tokens}")
                out.append(pool_tokens(h.float().cpu().numpy(), self.cfg.pooling))
        e = l2_normalize(np.concatenate(out).astype(np.float32))
        if e.shape[1] != 384:
            raise RuntimeError(f"embedding dim {e.shape[1]} != 384")
        return e

    def free(self) -> None:
        del self.model
        if self.device == "cuda":
            self.torch.cuda.empty_cache()
