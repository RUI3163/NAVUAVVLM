"""20 Hz pure-pursuit control loop. NEVER awaits the VLM, localizer or sim.

It reads: PlanBuffer snapshot, odometry state, latest depth (LatestSlot),
heartbeat. It writes: one velocity command into a LatestSlot (a separate
SimIO task actually talks to the simulator).
"""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from vlnav.odometry import DeadReckoning, OdomState
from vlnav.plan_buffer import Heartbeat, LatestSlot, Mode, PlanBuffer


@dataclass(frozen=True)
class VelocityCmd:
    vx: float = 0.0        # body forward
    vy: float = 0.0        # body left
    vz: float = 0.0        # up
    yaw_rate: float = 0.0  # rad/s, CCW


@dataclass
class ControllerConfig:
    rate_hz: float = 20.0
    lookahead_m: float = 6.0
    v_max: float = 4.0
    v_min: float = 0.8
    decel_gain: float = 0.6          # v <= gain * distance_to_end
    goal_tol_m: float = 1.0
    yaw_rate_max: float = math.radians(45)
    k_yaw: float = 1.5
    turn_in_place_rad: float = math.radians(60)
    target_alt_m: float = 10.0
    k_alt: float = 0.8
    vz_max: float = 1.5
    clearance_stop_m: float = 6.0
    clearance_slow_m: float = 15.0
    depth_roi: tuple[float, float, float, float] = (0.35, 0.65, 0.30, 0.70)  # rows, cols fractions
    depth_percentile: float = 10.0
    depth_max_m: float = 100.0
    depth_stale_s: float = 0.5
    heartbeat_timeout_s: float = 1.0
    sweep_rate: float = math.radians(30)


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def forward_clearance(depth: np.ndarray, cfg: ControllerConfig) -> float:
    d = np.asarray(depth, dtype=np.float32)
    d = np.where(np.isfinite(d) & (d > 0), d, cfg.depth_max_m)
    h, w = d.shape
    r0, r1, c0, c1 = cfg.depth_roi
    roi = d[int(r0 * h): max(int(r1 * h), int(r0 * h) + 1), int(c0 * w): max(int(c1 * w), int(c0 * w) + 1)]
    return float(np.percentile(roi, cfg.depth_percentile))


class PurePursuit:
    def __init__(self, lookahead: float) -> None:
        self.la = lookahead
        self._plan_id = -1
        self._seg = 0

    @staticmethod
    def _proj(p, a, b) -> tuple[float, tuple[float, float], float]:
        ax, ay = a
        bx, by = b
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 < 1e-9 else ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2
        tc = min(1.0, max(0.0, t))
        q = (ax + tc * dx, ay + tc * dy)
        return t, q, math.hypot(p[0] - q[0], p[1] - q[1])

    def target(self, x: float, y: float, wps, plan_id: int):
        """-> (target_point, remaining_distance_to_end)"""
        n = len(wps)
        if n == 0:
            return None, 0.0
        if n == 1:
            return wps[0], math.hypot(wps[0][0] - x, wps[0][1] - y)
        p = (x, y)
        if plan_id != self._plan_id:
            self._plan_id = plan_id
            self._seg = min(range(n - 1), key=lambda i: self._proj(p, wps[i], wps[i + 1])[2])
        t, q, _ = self._proj(p, wps[self._seg], wps[self._seg + 1])
        while t >= 1.0 and self._seg < n - 2:
            self._seg += 1
            t, q, _ = self._proj(p, wps[self._seg], wps[self._seg + 1])
        remaining = math.hypot(wps[self._seg + 1][0] - q[0], wps[self._seg + 1][1] - q[1])
        for i in range(self._seg + 1, n - 1):
            remaining += math.hypot(wps[i + 1][0] - wps[i][0], wps[i + 1][1] - wps[i][1])
        cur, left, i = q, self.la, self._seg
        while True:
            end = wps[i + 1]
            d = math.hypot(end[0] - cur[0], end[1] - cur[1])
            if d >= left:
                k = 0.0 if d < 1e-9 else left / d
                return (cur[0] + (end[0] - cur[0]) * k, cur[1] + (end[1] - cur[1]) * k), remaining
            left -= d
            cur, i = end, i + 1
            if i >= n - 1:
                return wps[-1], remaining


class ControlLoop:
    def __init__(self, cfg: ControllerConfig, plans: PlanBuffer, odom: DeadReckoning,
                 frame_slot: LatestSlot, cmd_slot: LatestSlot, heartbeat: Heartbeat,
                 clock=time.monotonic) -> None:
        self.cfg, self.plans, self.odom = cfg, plans, odom
        self.frames, self.cmd_slot, self.hb, self.clock = frame_slot, cmd_slot, heartbeat, clock
        self.pp = PurePursuit(cfg.lookahead_m)
        self.last_reason = "init"
        self.tick_dts: list[float] = []
        self.hover_ticks = 0
        self.ticks = 0

    def _vz(self, o: OdomState) -> float:
        c = self.cfg
        return float(np.clip(c.k_alt * (c.target_alt_m - o.alt), -c.vz_max, c.vz_max))

    def step(self, now: float) -> tuple[VelocityCmd, str]:
        """Pure, synchronous, non-blocking. Unit-testable."""
        c, o = self.cfg, self.odom.state()
        vz = self._vz(o)
        if self.hb.age(now) > c.heartbeat_timeout_s:
            return VelocityCmd(0, 0, vz, 0), "watchdog:mission_heartbeat"
        plan = self.plans.snapshot()
        if plan.mode is Mode.HOVER or not plan.waypoints:
            return VelocityCmd(0, 0, vz, 0), f"hover:{plan.reason}"
        if plan.mode is Mode.SWEEP:
            return VelocityCmd(0, 0, vz, c.sweep_rate), "sweep"
        tgt, remaining = self.pp.target(o.x, o.y, plan.waypoints, plan.plan_id)
        if tgt is None or remaining < c.goal_tol_m:
            return VelocityCmd(0, 0, vz, 0), "end_of_path"
        err = wrap_pi(math.atan2(tgt[1] - o.y, tgt[0] - o.x) - o.yaw)
        v = min(c.v_max, max(c.v_min, c.decel_gain * remaining))
        v *= max(0.0, math.cos(err)) if abs(err) < c.turn_in_place_rad else 0.0
        reason = "follow"
        frame, _ = self.frames.get()
        if frame is None or now - frame.t > c.depth_stale_s:
            v, reason = min(v, c.v_min), "follow:depth_stale"
        else:
            clr = forward_clearance(frame.depth, c)
            if clr < c.clearance_stop_m:
                v, reason = 0.0, "blocked:clearance"
            elif clr < c.clearance_slow_m:
                v *= (clr - c.clearance_stop_m) / (c.clearance_slow_m - c.clearance_stop_m)
                reason = "follow:slow"
        yaw_rate = float(np.clip(c.k_yaw * err, -c.yaw_rate_max, c.yaw_rate_max))
        return VelocityCmd(v, 0.0, vz, yaw_rate), reason

    async def run(self, stop: asyncio.Event) -> None:
        period = 1.0 / self.cfg.rate_hz
        nxt = self.clock()
        last = nxt
        while not stop.is_set():
            now = self.clock()
            cmd, self.last_reason = self.step(now)
            self.cmd_slot.put(cmd)
            self.ticks += 1
            self.hover_ticks += int(cmd.vx == 0 and cmd.yaw_rate == 0 and cmd.vy == 0)
            self.tick_dts.append(now - last)
            last = now
            nxt += period
            await asyncio.sleep(max(0.0, nxt - self.clock()))
            if self.clock() - nxt > 5 * period:
                nxt = self.clock()  # don't burst to catch up
