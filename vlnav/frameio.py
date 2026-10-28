"""The ONE image decode/save/load path. Mapping flights and live flights both use it, so offline and online
images are identical (RGB is saved lossless as PNG; depth as float32 metres in .npy)."""
from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np


def rgb_from_msg(msg, unpack_image=None, swap_rgb: bool = False) -> np.ndarray:
    """Project AirSim image message -> HxWx3 uint8 in OpenCV (BGR) order. Source: projectairsim.utils.unpack_image."""
    if unpack_image is None:
        from projectairsim.utils import unpack_image  # lazy: tests run without the sim client
    img = np.asarray(unpack_image(msg))
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        img = np.repeat(img.reshape(img.shape[0], img.shape[1], 1), 3, axis=2)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    if swap_rgb:
        img = img[:, :, ::-1]
    return np.ascontiguousarray(img, dtype=np.uint8)


def depth_from_array(arr: np.ndarray, max_m: float = 100.0) -> np.ndarray:
    """16FC1 metres (fp16) -> float32 HxW. Non-finite AND > max_m -> NaN (no hit). Measured in M0: sky pixels are
    finite (~16416 m), not +inf, so a finite-only mask is not enough."""
    d = np.asarray(arr, dtype=np.float32)
    if d.ndim == 3:
        d = d[:, :, 0]
    d = d.copy()
    d[~np.isfinite(d) | (d > max_m)] = np.nan
    return d


def depth_from_msg(msg, unpack_image=None, max_m: float = 100.0) -> np.ndarray:
    if unpack_image is None:
        from projectairsim.utils import unpack_image
    return depth_from_array(unpack_image(msg), max_m)


def depth_vis(depth: np.ndarray, max_m: float = 100.0) -> np.ndarray:
    d = np.nan_to_num(depth, nan=max_m)
    return cv2.applyColorMap((np.clip(d / max_m, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)


def save_frame(dirpath: str, stem: str, bgr: np.ndarray, depth: Optional[np.ndarray] = None) -> dict:
    os.makedirs(dirpath, exist_ok=True)
    paths = {"rgb": os.path.join(dirpath, f"{stem}_rgb.png")}
    if not cv2.imwrite(paths["rgb"], bgr):
        raise IOError(f"could not write {paths['rgb']}")
    if depth is not None:
        paths["depth"] = os.path.join(dirpath, f"{stem}_depth.npy")
        np.save(paths["depth"], depth.astype(np.float32))
        paths["depth_vis"] = os.path.join(dirpath, f"{stem}_depth_vis.png")
        cv2.imwrite(paths["depth_vis"], depth_vis(depth))
    return paths


def load_frame(dirpath: str, stem: str):
    bgr = cv2.imread(os.path.join(dirpath, f"{stem}_rgb.png"), cv2.IMREAD_COLOR)
    dp = os.path.join(dirpath, f"{stem}_depth.npy")
    return bgr, (np.load(dp) if os.path.isfile(dp) else None)
