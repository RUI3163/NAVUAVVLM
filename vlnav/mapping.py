"""Keyframe policy (pure). A keyframe is taken every `dist_m` of odometry travel or `yaw_deg` of odometry yaw."""
from __future__ import annotations

import math
from typing import Optional, Tuple


class KeyframePolicy:
    def __init__(self, dist_m: float, yaw_deg: float):
        self.dist_m, self.yaw_deg = dist_m, yaw_deg
        self._last: Optional[Tuple[float, float, float]] = None

    def update(self, n: float, e: float, yaw_rad: float) -> bool:
        if self._last is not None:
            ln, le, ly = self._last
            dyaw = abs((math.degrees(yaw_rad - ly) + 180.0) % 360.0 - 180.0)
            if math.hypot(n - ln, e - le) < self.dist_m and dyaw < self.yaw_deg:
                return False
        self._last = (n, e, yaw_rad)
        return True
