"""Versioned plan buffer. The mission manager is the ONLY writer; the control loop only reads `latest()`, which returns
an immutable snapshot (plan_id and waypoints always belong together) and never blocks."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

XY = Tuple[float, float]


@dataclass(frozen=True)
class Plan:
    plan_id: int
    waypoints: Tuple[XY, ...]
    goal_xy: Optional[XY] = None
    created: float = 0.0


class PlanBuffer:
    def __init__(self):
        self._plan = Plan(0, ())
        self._w = threading.Lock()                     # serialises writers only; readers take no lock

    def publish(self, waypoints: Sequence[XY], goal_xy: Optional[XY] = None, now: Optional[float] = None) -> int:
        with self._w:
            p = Plan(self._plan.plan_id + 1, tuple((float(a), float(b)) for a, b in waypoints), goal_xy,
                     time.monotonic() if now is None else now)
            self._plan = p                             # single reference assignment = atomic swap
            return p.plan_id

    def latest(self) -> Plan:
        return self._plan
