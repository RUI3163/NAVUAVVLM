"""Phase-A scripted mapping flights (DATA COLLECTION, not navigation).
STEERING uses ground truth + the surveyed landmarks so the passes follow the road precisely. What is STORED for each
keyframe is (a) the camera image via frameio.save_frame (same path as live flights), (b) the GPS-denied odometry pose
(odometry.DeadReckoner: commanded velocity + gyro yaw + injected noise, never the sim pose), and (c) the ground-truth
pose as a log-only field. This module is on the ground-truth allow-list in tests/test_no_groundtruth_in_nav.py.

One pass = round trip: take off at P0 -> side-step to the road -> north past X1 -> turn 180 -> south along the road
-> back to P0 -> face north -> land.  Odometry origin = P0 for every pass, so all roots share one odometry frame."""
from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from typing import Dict, List

from .config import Config
from .frameio import save_frame
from .groundtruth import GroundTruth
from .mapping import KeyframePolicy
from .odometry import DeadReckoner
from .sim_interface import SimInterface

log = logging.getLogger("mapping_flight")


def _wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class PassFlyer:
    def __init__(self, sim: SimInterface, gt: GroundTruth, cfg: Config, lm: Dict, pass_idx: int, out_dir: str):
        self.sim, self.gt, self.cfg, self.lm, self.k, self.out = sim, gt, cfg, lm, pass_idx, out_dir
        self.m = cfg.map
        self.p0 = (lm["p0"]["n"], lm["p0"]["e"])
        cl = lm["centreline"]
        self.e_road = sum(p[1] for p in cl) / len(cl) + self.m.pass_lateral_offsets_m[pass_idx % len(self.m.pass_lateral_offsets_m)]
        self.n_south, self.n_north = cl[0][0], lm["x1"]["n"] + self.m.overshoot_m
        self.odo = DeadReckoner(cfg.odom, cfg.seed + 1000 * pass_idx, yaw0_rad=gt.yaw)
        sim.subscribe_gyro(self.odo.on_gyro)
        self.policy = KeyframePolicy(self.m.keyframe_dist_m, self.m.keyframe_yaw_deg)
        self.nodes: List[dict] = []
        self.yaw_sign = 1.0
        self.alt = cfg.flight.takeoff_alt_m

    # ---- primitives ----
    async def _step(self, vn: float, ve: float, yaw_rate: float = 0.0) -> None:
        gt = self.gt
        vd = _clip(0.7 * ((gt.ground_z0 - self.alt) - gt.z), -2.0, 2.0)
        t0 = time.monotonic()
        await self.sim.velocity(vn, ve, vd, self.m.step_s, yaw_rate=yaw_rate)
        self.odo.update(vn, ve, time.monotonic() - t0)       # integrate over the time the command was active

    async def _keyframe(self) -> None:
        n, e, yaw = self.odo.pose()
        if not self.policy.update(n, e, yaw):
            return
        rgb, _ = await asyncio.to_thread(self.sim.get_frame, False)
        idx = len(self.nodes)
        stem = f"p{self.k}_{idx:04d}"
        paths = save_frame(os.path.join(self.out, "images"), stem, rgb)
        g = self.gt
        self.nodes.append({"root": self.k, "idx": idx, "t": time.time(), "odo": [n, e, yaw],
                           "gt": [g.n - self.p0[0], g.e - self.p0[1], g.yaw],        # LOG ONLY
                           "image": os.path.relpath(paths["rgb"], self.out)})

    async def climb(self) -> None:
        t0 = time.monotonic()
        while abs(self.gt.altitude - self.alt) > self.cfg.flight.alt_tol_m:
            await self._step(0.0, 0.0)
            if time.monotonic() - t0 > self.cfg.flight.climb_timeout_s:
                raise RuntimeError("climb timeout")

    async def goto(self, n: float, e: float, record: bool, tol: float = 1.0, timeout: float = 120.0) -> None:
        t0 = time.monotonic()
        while True:
            dn, de = n - self.gt.n, e - self.gt.e
            d = math.hypot(dn, de)
            if d < tol:
                return
            sp = min(self.m.cruise_mps, max(0.8, 0.6 * d))
            await self._step(dn / d * sp, de / d * sp)
            if record:
                await self._keyframe()
            if time.monotonic() - t0 > timeout:
                raise RuntimeError(f"goto ({n:.1f},{e:.1f}) timeout, {d:.1f} m left")

    async def fly_line(self, n_target: float, record: bool, timeout: float = 150.0) -> None:
        t0 = time.monotonic()
        while True:
            rem = n_target - self.gt.n
            if abs(rem) < 0.6:
                return
            sp = min(self.m.cruise_mps, max(0.8, 0.6 * abs(rem)))
            await self._step(math.copysign(sp, rem), _clip(1.0 * (self.e_road - self.gt.e), -1.0, 1.0))
            if record:
                await self._keyframe()
            if time.monotonic() - t0 > timeout:
                raise RuntimeError("fly_line timeout")

    async def turn_to(self, yaw_deg: float, record: bool, timeout: float = 40.0) -> None:
        """Rotate in place. Direction sign is auto-checked: if the error grows during the first 1.5 s the yaw-rate sign
        convention is inverted and is flipped (logged)."""
        t0, err0, checked = time.monotonic(), None, False
        while True:
            err = _wrap180(yaw_deg - math.degrees(self.gt.yaw))
            if abs(err) < 5.0:
                return
            err0 = abs(err) if err0 is None else err0
            if not checked and time.monotonic() - t0 > 1.5:
                checked = True
                if abs(err) > err0 + 5.0:
                    self.yaw_sign = -self.yaw_sign
                    log.warning("yaw-rate sign inverted -> flipped to %+.0f", self.yaw_sign)
            rate = math.copysign(math.radians(self.m.turn_rate_dps), err) * self.yaw_sign
            await self._step(0.0, 0.0, yaw_rate=rate)
            if record:
                await self._keyframe()
            if time.monotonic() - t0 > timeout:
                raise RuntimeError(f"turn_to({yaw_deg}) timeout, err {err:.0f} deg")

    # ---- the pass ----
    async def run(self) -> List[dict]:
        sim, rec_turn = self.sim, self.m.record_turn
        log.info("pass %d: e_road=%.2f, N %.1f..%.1f", self.k, self.e_road, self.n_south, self.n_north)
        await sim.takeoff()
        await self.climb()
        await self.goto(self.n_south, self.e_road, record=False)
        await self.fly_line(self.n_north, record=True)
        await self.turn_to(180.0, record=rec_turn)
        await self.fly_line(self.n_south, record=True)
        await self.goto(self.p0[0], self.p0[1], record=False)
        await self.turn_to(0.0, record=False)
        await sim.hover()
        n, e, _ = self.odo.pose()
        log.info("pass %d done: %d keyframes; odometry end (%.1f, %.1f) vs true (%.1f, %.1f)", self.k,
                 len(self.nodes), n, e, self.gt.n - self.p0[0], self.gt.e - self.p0[1])
        await sim.land(lambda: self.gt.altitude)
        return self.nodes
