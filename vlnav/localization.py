"""Visual localisation against the topological map (navigation-side: embeddings + odometry only, NO ground truth).

Per frame:  cosine kNN over node embeddings, GATED by odometry uncertainty (a node is a candidate only if it lies
within  floor + k*sigma  of the predicted position, sigma growing with distance flown since the last fix and with the
node's own mapping drift), then a SEQUENCE VOTE: the last `vote_frames` top-1 matches, transported to the current
frame with relative odometry, must agree within a spread tolerance before a fix is accepted.
An accepted fix re-anchors the odometry (offset) and resets the uncertainty."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Sequence, Tuple

import numpy as np

from .config import LocCfg


@dataclass
class Fix:
    index: int                      # row in the map arrays
    node_id: int
    xy: Tuple[float, float]         # fused position in the map/odometry frame
    spread_m: float
    sim: float


@dataclass
class LocResult:
    fix: Optional[Fix]
    top1: Optional[int]             # gated top-1 row
    raw_top1: int                   # UNGATED top-1 row (diagnostic only)
    n_gated: int
    gate_m: float                   # gate radius for a node with no mapping drift
    sigma_m: float                  # live position sigma
    lost: bool


def cumulative_odo_distance(xy: np.ndarray, root: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Odometry path length from the start of each node's own pass (its mapping-drift exposure)."""
    d = np.zeros(len(xy))
    for r in np.unique(root):
        rows = np.where(root == r)[0]
        rows = rows[np.argsort(idx[rows])]
        seg = np.linalg.norm(np.diff(xy[rows], axis=0), axis=1)
        d[rows] = np.concatenate([[0.0], np.cumsum(seg)])
    return d


class Localizer:
    def __init__(self, emb: np.ndarray, xy: np.ndarray, root: np.ndarray, idx: np.ndarray, ids: Sequence[int],
                 cfg: LocCfg, spacing_m: float):
        self.E = np.asarray(emb, np.float32)
        self.xy = np.asarray(xy, np.float64)
        self.ids = np.asarray(ids)
        self.cfg, self.spacing = cfg, float(spacing_m)
        self.node_dist = cumulative_odo_distance(self.xy, np.asarray(root), np.asarray(idx))
        self.reset()

    def reset(self) -> None:
        self.offset = np.zeros(2)
        self.dist_since_fix = 0.0
        self.fixed = False
        self._last_odo: Optional[np.ndarray] = None
        self._hist: Deque[Tuple[int, np.ndarray, float]] = deque(maxlen=self.cfg.vote_frames)
        self.frames_since_fix = 0

    def sigma(self) -> float:
        base = self.cfg.sigma_fix_factor * self.spacing if self.fixed else self.cfg.sigma_start_m
        return base + self.cfg.drift_frac * self.dist_since_fix

    def step(self, emb: np.ndarray, odo_xy: Sequence[float]) -> LocResult:
        c = self.cfg
        odo = np.asarray(odo_xy, np.float64)
        if self._last_odo is not None:
            self.dist_since_fix += float(np.linalg.norm(odo - self._last_odo))
        self._last_odo = odo
        pred = odo + self.offset
        sims = self.E @ np.asarray(emb, np.float32)
        raw = int(np.argmax(sims))
        sig = self.sigma()
        floor = c.gate_floor_factor * self.spacing
        node_sig = np.sqrt(sig ** 2 + (c.drift_frac * self.node_dist) ** 2)
        gate = floor + c.gate_sigma_k * node_sig
        mask = np.linalg.norm(self.xy - pred, axis=1) <= gate
        self.frames_since_fix += 1
        lost = self.frames_since_fix > c.lost_after_frames
        res = LocResult(None, None, raw, int(mask.sum()), floor + c.gate_sigma_k * sig, sig, lost)
        if not mask.any():
            return res
        cand = np.where(mask)[0]
        cand = cand[np.argsort(-sims[cand])][: c.top_k]
        top1 = int(cand[0])
        res.top1 = top1
        self._hist.append((top1, odo.copy(), float(sims[top1])))
        if len(self._hist) < c.vote_frames:
            return res
        P = np.array([self.xy[i] + (odo - o) for i, o, _ in self._hist])      # votes transported to "now"
        spread = float(np.max(np.linalg.norm(P[:, None] - P[None], axis=-1)))
        if spread > c.vote_spread_factor * self.spacing:
            return res
        m = np.median(P, axis=0)
        node = int(cand[np.argmin(np.linalg.norm(self.xy[cand] - m, axis=1))])  # appearance-gated, vote-consistent
        self.offset = m - odo
        self.dist_since_fix, self.fixed, self.frames_since_fix = 0.0, True, 0
        res.fix = Fix(node, int(self.ids[node]), (float(m[0]), float(m[1])), spread, float(sims[node]))
        res.lost = False
        return res
