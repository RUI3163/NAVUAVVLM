"""Pure control maths for the 20 Hz loop (no I/O, no simulator, no awaits).
Frames: positions are the localiser-corrected odometry position in the MAP frame (north, east)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from .config import ControlCfg

XY = Tuple[float, float]


@dataclass
class PPResult:
    vn: float
    ve: float
    speed: float
    target: XY
    seg: int              # index of the path segment the vehicle is on (monotonic: never goes backwards)
    dist_to_end: float
    done: bool


def _project(p: XY, a: XY, b: XY) -> Tuple[float, float]:
    dx, dy = b[0] - a[0], b[1] - a[1]
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 < 1e-12 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2))
    return t, math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy))


def pure_pursuit(pos: XY, wps: Sequence[XY], cfg: ControlCfg, seg_hint: int = 0) -> PPResult:
    if len(wps) == 0:
        return PPResult(0.0, 0.0, 0.0, pos, 0, 0.0, True)
    if len(wps) == 1:
        wps = [pos, wps[0]]
    nseg = len(wps) - 1
    best, seg, tbest = 1e18, seg_hint, 0.0
    for i in range(min(seg_hint, nseg - 1), nseg):               # only forward from the hint: no backtracking
        t, d = _project(pos, wps[i], wps[i + 1])
        if d < best - 1e-9:
            best, seg, tbest = d, i, t
    a, b = wps[seg], wps[seg + 1]
    segl = math.hypot(b[0] - a[0], b[1] - a[1])
    remain = cfg.lookahead_m + tbest * segl                      # arclength measured from the segment start
    i, tgt = seg, None
    while True:
        a, b = wps[i], wps[i + 1]
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        if remain <= L or i == nseg - 1:
            f = 1.0 if L < 1e-9 else min(1.0, remain / L)
            tgt = (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]))
            break
        remain -= L
        i += 1
    end = wps[-1]
    d_end = math.hypot(end[0] - pos[0], end[1] - pos[1])
    if d_end < cfg.arrive_tol_m:
        return PPResult(0.0, 0.0, 0.0, end, seg, d_end, True)
    dn, de = tgt[0] - pos[0], tgt[1] - pos[1]
    n = math.hypot(dn, de)
    speed = min(cfg.cruise_mps, max(cfg.min_speed_mps, cfg.end_slow_gain * d_end))
    return PPResult(dn / n * speed if n > 1e-9 else 0.0, de / n * speed if n > 1e-9 else 0.0, speed, tgt, seg, d_end, False)


def forward_clearance(depth: np.ndarray, cfg: ControlCfg) -> float:
    """Distance (m) to the nearest obstacle in the central ROI. NaN (sky / no hit) counts as free space.
    A LOW PERCENTILE is used, so features covering less than ~clear_pct % of the ROI (thin wires) are not seen."""
    h, w = depth.shape[:2]
    rh, rw = int(h * cfg.roi_h_frac / 2), int(w * cfg.roi_w_frac / 2)
    roi = depth[h // 2 - rh: h // 2 + rh, w // 2 - rw: w // 2 + rw]
    v = roi[np.isfinite(roi)]
    return float("inf") if v.size == 0 else float(np.percentile(v, cfg.clear_pct))


def clearance_scale(clear_m: float, cfg: ControlCfg) -> float:
    if clear_m >= cfg.clear_slow_m:
        return 1.0
    if clear_m <= cfg.clear_stop_m:
        return 0.0
    return (clear_m - cfg.clear_stop_m) / (cfg.clear_slow_m - cfg.clear_stop_m)


def apply_clearance(vn: float, ve: float, yaw_rad: float, clear_m: float, cfg: ControlCfg) -> Tuple[float, float, float]:
    """Scale the command only when it points roughly along the camera axis (a sideways move is not covered by the
    forward-looking depth image). Returns (vn, ve, scale)."""
    sp = math.hypot(vn, ve)
    if sp < 1e-6:
        return vn, ve, 1.0
    cos = (vn * math.cos(yaw_rad) + ve * math.sin(yaw_rad)) / sp
    if cos < cfg.forward_cone_cos:
        return vn, ve, 1.0
    s = clearance_scale(clear_m, cfg)
    return vn * s, ve * s, s
