"""Watchdogs (pure, time injected so they are testable)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .config import RecoveryCfg


class Heartbeat:
    """Mission-manager heartbeat: the control loop hovers when it is older than `timeout_s`."""
    def __init__(self, timeout_s: float, t0: float = 0.0):
        self.timeout_s, self._last = timeout_s, t0

    def beat(self, t: float) -> None:
        self._last = t

    def alive(self, t: float) -> bool:
        return (t - self._last) <= self.timeout_s


@dataclass
class Action:
    kind: str                 # fly | hover | scan | abort
    yaw_rate: float = 0.0


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class LostRecovery:
    """On lost localisation: hover and yaw 360 deg, then wait for a fix; repeat up to max_attempts, then abort (hover)."""
    def __init__(self, cfg: RecoveryCfg):
        self.cfg = cfg
        self.mode = "track"
        self.attempts = 0
        self._swept = 0.0
        self._last_yaw = 0.0
        self._t_reacq = 0.0

    def _start_scan(self, yaw: float) -> Action:
        self.mode, self._swept, self._last_yaw = "scan", 0.0, yaw
        self.attempts += 1
        return Action("scan", self.cfg.scan_rate_rad_s)

    def update(self, t: float, lost: bool, yaw_rad: float) -> Action:
        if self.mode == "track":
            return self._start_scan(yaw_rad) if lost else Action("fly")
        if self.mode == "scan":
            self._swept += abs(_wrap(yaw_rad - self._last_yaw)); self._last_yaw = yaw_rad
            if not lost:
                self.mode, self.attempts = "track", 0
                return Action("fly")
            if self._swept >= 2 * math.pi:
                self.mode, self._t_reacq = "reacquire", t
                return Action("hover")
            return Action("scan", self.cfg.scan_rate_rad_s)
        if self.mode == "reacquire":
            if not lost:
                self.mode, self.attempts = "track", 0
                return Action("fly")
            if t - self._t_reacq >= self.cfg.reacquire_s:
                if self.attempts >= self.cfg.max_attempts:
                    self.mode = "abort"
                    return Action("abort")
                return self._start_scan(yaw_rad)
            return Action("hover")
        return Action("abort")
