"""GROUND-TRUTH pose (sim actual_pose). Evaluator / logger / survey ONLY -- navigation modules must never import
this module (enforced by tests/test_no_groundtruth_in_nav.py). Subscription pattern from the reference repo."""
from __future__ import annotations

import math
import threading
from typing import Optional, Tuple


def quat_to_yaw(w: float, x: float, y: float, z: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class GroundTruth:
    def __init__(self, sim, ground_z_world: Optional[float] = None):
        self._lock = threading.Lock()
        self._seen = threading.Event()
        self._ground_z_override = ground_z_world
        self.spawn: Optional[Tuple[float, float, float]] = None   # (x, y, z) NED at first message
        self.ground_z0 = 0.0
        self.n = self.e = self.z = self.yaw = 0.0
        sim.client.subscribe(sim.drone.robot_info["actual_pose"], self._on_pose)
        sim._log('client.subscribe(drone.robot_info["actual_pose"], cb)')

    def _on_pose(self, _topic, msg) -> None:
        p, q = msg["position"], msg["orientation"]
        with self._lock:
            if not self._seen.is_set():
                self.spawn = (float(p["x"]), float(p["y"]), float(p["z"]))
                self.ground_z0 = float(p["z"]) if self._ground_z_override is None else self._ground_z_override
                self._seen.set()
            self.n, self.e, self.z = float(p["x"]), float(p["y"]), float(p["z"])
            self.yaw = quat_to_yaw(q["w"], q["x"], q["y"], q["z"])

    def wait(self, timeout: float) -> bool:
        return self._seen.wait(timeout)

    @property
    def altitude(self) -> float:
        with self._lock:
            return self.ground_z0 - self.z
