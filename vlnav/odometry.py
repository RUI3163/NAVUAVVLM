"""GPS-denied odometry (navigation-side; no ground truth, no sim pose).
Position: integrate the COMMANDED world-NED velocity, corrupted by a per-flight seeded speed-scale error and a
constant heading bias (rotation of the velocity vector). Yaw: integrate the IMU gyro z-rate, plus the same bias.
Limitation (documented): commanded != achieved velocity, so real motion differs from the estimate by control lag;
M2 logs ground truth next to it to measure the resulting drift."""
from __future__ import annotations

import math
import threading
from typing import Optional, Tuple

import numpy as np

from .config import OdomCfg


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class DeadReckoner:
    def __init__(self, cfg: OdomCfg, seed: int, yaw0_rad: float = 0.0):
        rng = np.random.default_rng(seed)
        self.cfg = cfg
        self.scale = 1.0 + float(rng.normal(0.0, cfg.scale_sigma))
        self.bias = math.radians(float(rng.normal(0.0, cfg.heading_bias_sigma_deg)))
        self.n = self.e = self.dist = 0.0
        self._yaw = yaw0_rad
        self._last_t: Optional[float] = None
        self._lock = threading.Lock()

    def update(self, vn_cmd: float, ve_cmd: float, dt: float) -> None:
        c, s = math.cos(self.bias), math.sin(self.bias)
        vn = self.scale * (c * vn_cmd - s * ve_cmd)
        ve = self.scale * (s * vn_cmd + c * ve_cmd)
        with self._lock:
            self.n += vn * dt
            self.e += ve * dt
            self.dist += math.hypot(vn, ve) * dt

    def on_gyro(self, wz: float, t: float) -> None:
        with self._lock:
            if self._last_t is not None:
                self._yaw = wrap_pi(self._yaw + self.cfg.gyro_yaw_sign * wz * min(max(t - self._last_t, 0.0), 0.1))
            self._last_t = t

    def pose(self) -> Tuple[float, float, float]:
        """(north, east, yaw_rad) in the odometry frame: origin = start position, axes = NED."""
        with self._lock:
            return self.n, self.e, wrap_pi(self._yaw + self.bias)
