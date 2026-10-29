"""Perception worker (~2-5 Hz): latest frame only -> DINOv2 -> gated kNN fix."""
from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional, Protocol

import numpy as np

from vlnav.odometry import DeadReckoning
from vlnav.plan_buffer import LatestSlot


@dataclass
class Frame:
    frame_id: int
    t: float
    rgb: np.ndarray
    depth: np.ndarray


@dataclass(frozen=True)
class LocFix:
    frame_id: int
    node: Optional[Any]
    status: str          # "fix" | "searching" | "lost"
    sigma_m: float
    t: float


class Localizer(Protocol):
    def localize(self, rgb: np.ndarray, odom_xy_yaw: tuple[float, float, float]) -> tuple[Optional[Any], str, float]:
        """-> (node_or_None, status, sigma_m). Wrap the M3 sequence-voted gated kNN here."""


@dataclass
class PerceptionConfig:
    rate_hz: float = 4.0


class PerceptionWorker:
    def __init__(self, cfg: PerceptionConfig, localizer: Localizer, frame_slot: LatestSlot,
                 odom: DeadReckoning, fix_slot: LatestSlot, clock=time.monotonic) -> None:
        self.cfg, self.loc, self.frames, self.odom, self.fixes, self.clock = cfg, localizer, frame_slot, odom, fix_slot, clock
        self._ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="perception")
        self.latencies: list[float] = []

    def _work(self, frame: Frame) -> LocFix:
        t0 = self.clock()
        o = self.odom.state()
        node, status, sigma = self.loc.localize(frame.rgb, (o.x, o.y, o.yaw))
        self.latencies.append(self.clock() - t0)
        return LocFix(frame.frame_id, node, status, sigma, self.clock())

    async def run(self, stop: asyncio.Event) -> None:
        loop = asyncio.get_running_loop()
        seen = 0
        period = 1.0 / self.cfg.rate_hz
        while not stop.is_set():
            t0 = self.clock()
            frame, seen = self.frames.take_if_newer(seen)  # stale queue is never consumed
            if frame is not None:
                self.fixes.put(await loop.run_in_executor(self._ex, self._work, frame))
            await asyncio.sleep(max(0.01, period - (self.clock() - t0)))
        self._ex.shutdown(wait=False)
