"""Offline mock: kinematic unicycle on a straight road. `true_xy` exists ONLY here
(for the mock localizer / eval); navigation modules never see it."""
from __future__ import annotations

import math
import time
from typing import Any, Optional

import numpy as np

from vlnav.controller import VelocityCmd
from vlnav.odometry import ImuSample
from vlnav.perception import Frame


class MockSim:
    def __init__(self, x0: float = 0.0, y0: float = 0.0, yaw0: float = 0.0,
                 intersection_x: float = 108.5, alt: float = 10.0) -> None:
        self.x, self.y, self.yaw, self.alt = x0, y0, yaw0, alt
        self.ix = intersection_x
        self._cmd = VelocityCmd()
        self._t = time.monotonic()
        self._fid = 0

    @property
    def true_xy(self) -> tuple[float, float]:
        return self.x, self.y

    def _adv(self) -> None:
        now = time.monotonic()
        dt, self._t = now - self._t, now
        c = self._cmd
        self.yaw += c.yaw_rate * dt
        cs, sn = math.cos(self.yaw), math.sin(self.yaw)
        self.x += (cs * c.vx - sn * c.vy) * dt
        self.y += (sn * c.vx + cs * c.vy) * dt
        self.alt += c.vz * dt

    async def imu(self) -> ImuSample:
        self._adv()
        c = self._cmd
        return ImuSample(c.vx, c.vy, c.yaw_rate, self.alt, time.monotonic())

    async def send_velocity(self, cmd: VelocityCmd) -> None:
        self._adv()
        self._cmd = cmd

    async def frame(self) -> Optional[Frame]:
        self._adv()
        self._fid += 1
        near = abs(self.x - self.ix) < 12.0
        depth = np.full((48, 64), 200.0, np.float32)
        depth[:, :16] = depth[:, -16:] = 60.0 if near else 12.0  # lateral openness
        return Frame(self._fid, time.monotonic(), np.zeros((48, 64, 3), np.uint8), depth)


class MockLocalizer:
    """Nearest-node 'fix' from the mock's true pose, with a configurable dropout window."""

    def __init__(self, sim: MockSim, g, dropout: range = range(0), sigma: float = 3.0) -> None:
        from vlnav.graph_utils import node_xy
        self.sim, self.dropout, self.sigma, self.n = sim, dropout, sigma, 0
        self.xy = {n: node_xy(g, n) for n in g.nodes}
        self.miss = 0

    def localize(self, rgb, odom_xy_yaw) -> tuple[Any, str, float]:
        self.n += 1
        if self.n in self.dropout:
            self.miss += 1
            return None, ("lost" if self.miss >= 8 else "searching"), self.sigma
        self.miss = 0
        x, y = self.sim.true_xy
        return min(self.xy, key=lambda k: (self.xy[k][0] - x) ** 2 + (self.xy[k][1] - y) ** 2), "fix", self.sigma
