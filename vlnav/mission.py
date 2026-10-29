"""Mission manager: sole owner/writer of the graph; A* routing; arrival logic;
lost-recovery. `tick()` is synchronous and non-blocking (VLM is polled, never awaited)."""
from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import networkx as nx

from vlnav.graph_utils import edge_cost, node_xy
from vlnav.odometry import DeadReckoning
from vlnav.perception import LocFix
from vlnav.plan_buffer import Heartbeat, LatestSlot, Mode, PlanBuffer
from vlnav.vlm_worker import VLMClient, VLMKind

log = logging.getLogger("vlnav.mission")


class State(str, Enum):
    NAVIGATE = "navigate"
    VERIFY = "verify"
    LOST_SWEEP = "lost_sweep"
    ARRIVED = "arrived"
    FAILED = "failed"


@dataclass
class MissionConfig:
    tick_hz: float = 10.0
    arrival_hits: int = 3
    arrival_hops: int = 1
    arrival_neighbor_radius_m: float = 8.0   # = M3 match_radius_m; excludes shortcut hops
    arrival_dist_m: float = 5.0
    lost_frames: int = 8
    sweep_rate_rad_s: float = math.radians(30)
    max_sweeps: int = 3
    verify_timeout_s: float = 6.0
    verify_cooldown_s: float = 3.0
    reanchor_gain: float = 0.3
    route_deviation_m: float = 8.0           # replan only if the fix is farther than this from the route
    mission_timeout_s: float = 300.0
    caption_nodes: bool = True


def plan_route(g: nx.Graph, src: Any, dst: Any) -> list[Any]:
    """A* over the topological graph (shortcut edges included)."""
    def h(u, v):
        (x1, y1), (x2, y2) = node_xy(g, u), node_xy(g, v)
        return math.hypot(x2 - x1, y2 - y1)
    return nx.astar_path(g, src, dst, heuristic=h, weight=lambda u, v, d: edge_cost(g, u, v, d))


class MissionManager:
    def __init__(self, g: nx.Graph, start_node: Any, goal_node: Any, plans: PlanBuffer, hb: Heartbeat,
                 fix_slot: LatestSlot, frame_slot: LatestSlot, odom: DeadReckoning, vlm: VLMClient,
                 cfg: MissionConfig, clock=time.monotonic) -> None:
        self.g, self.start_node, self.goal = g, start_node, goal_node
        self.plans, self.hb, self.fixes, self.frames = plans, hb, fix_slot, frame_slot
        self.odom, self.vlm, self.cfg, self.clock = odom, vlm, cfg, clock
        self.state = State.NAVIGATE
        self.reason = ""
        self.t0: Optional[float] = None
        self.hits = 0
        self.no_fix = 0
        self.route: list[Any] = []
        self.cur_node: Any = start_node
        self._last_frame = -1
        self._verify_id: Optional[int] = None
        self._verify_t = 0.0
        self._cooldown_until = 0.0
        self._sweep_t = 0.0
        self.sweeps = 0
        self._captioned: set = set()
        self._cap_ids: dict[int, Any] = {}
        self.events: list[tuple[float, str]] = []
        self.verify_log: list[dict] = []

    # ---- helpers
    def _ev(self, now: float, msg: str) -> None:
        self.events.append((now, msg))
        log.info("[%.2f] %s", now, msg)

    def _accept_set(self) -> set:
        s = {self.goal}
        if self.cfg.arrival_hops >= 1:
            gx, gy = node_xy(self.g, self.goal)
            for n in self.g.neighbors(self.goal):
                if self.g.edges[self.goal, n].get("kind") == "shortcut":
                    continue  # shortcuts skip nodes: not a true 1-hop neighbour
                x, y = node_xy(self.g, n)
                if math.hypot(x - gx, y - gy) <= self.cfg.arrival_neighbor_radius_m:
                    s.add(n)
        return s

    def _off_route(self, node: Any) -> bool:
        if node in self.route:
            return False
        x, y = node_xy(self.g, node)
        return all(math.hypot(x - rx, y - ry) > self.cfg.route_deviation_m
                   for rx, ry in (node_xy(self.g, n) for n in self.route)) if self.route else True

    def _dist_goal(self) -> float:
        o = self.odom.state()
        gx, gy = node_xy(self.g, self.goal)
        return math.hypot(o.x - gx, o.y - gy)

    def _publish_route(self, src: Any, reason: str) -> None:
        self.route = plan_route(self.g, src, self.goal)
        self.plans.publish([node_xy(self.g, n) for n in self.route], Mode.FOLLOW, reason)

    def _fail(self, now: float, why: str) -> None:
        self.state, self.reason = State.FAILED, why
        self.plans.publish([], Mode.HOVER, f"failed:{why}")
        self._ev(now, f"FAILED {why}")

    # ---- lifecycle
    def start(self, now: Optional[float] = None) -> None:
        now = self.clock() if now is None else now
        self.t0 = now
        self.hb.beat(now)
        self._publish_route(self.start_node, "initial")
        self._ev(now, f"route {self.route}")

    @property
    def done(self) -> bool:
        return self.state in (State.ARRIVED, State.FAILED)

    @property
    def success(self) -> bool:
        return self.state is State.ARRIVED

    def tick(self, now: Optional[float] = None) -> None:
        now = self.clock() if now is None else now
        self.hb.beat(now)
        if self.done:
            return
        if self.t0 is not None and now - self.t0 > self.cfg.mission_timeout_s:
            return self._fail(now, "timeout")
        self._apply_vlm(now)
        fix, _ = self.fixes.get()
        if fix is not None and fix.frame_id > self._last_frame:
            self._last_frame = fix.frame_id
            self._on_fix(fix, now)
        if self.state is State.NAVIGATE:
            self._maybe_verify(now)
        elif self.state is State.VERIFY and now - self._verify_t > self.cfg.verify_timeout_s:
            self._verify_done(now, self._fallback_answer(), "timeout->rule")
        elif self.state is State.LOST_SWEEP:
            self._sweep_logic(now)

    # ---- fixes
    def _on_fix(self, fix: LocFix, now: float) -> None:
        ok = fix.status == "fix" and fix.node is not None
        self.no_fix = 0 if ok else self.no_fix + 1
        if ok:
            self.cur_node = fix.node
            gx, gy = node_xy(self.g, fix.node)
            self.odom.reanchor(gx, gy, self.cfg.reanchor_gain)
            self.hits = self.hits + 1 if fix.node in self._accept_set() else 0
            if self.state is State.NAVIGATE and self._off_route(fix.node):
                self._publish_route(fix.node, f"replan@{fix.node}")
                self._ev(now, f"replan from {fix.node}: {self.route}")
            if self.cfg.caption_nodes and fix.node not in self._captioned:
                self._captioned.add(fix.node)
                f, _ = self.frames.get()
                if f is not None:
                    self._cap_ids[self.vlm.submit(VLMKind.CAPTION, {"rgb": f.rgb}, key=f"cap{fix.node}")] = fix.node
            if self.state is State.LOST_SWEEP:
                self.state = State.NAVIGATE
                self._publish_route(fix.node, "reacquired")
                self._ev(now, f"re-anchored at {fix.node}")
        else:
            self.hits = 0
        if self.state is State.NAVIGATE and (fix.status == "lost" or self.no_fix >= self.cfg.lost_frames):
            self.state, self._sweep_t, self.sweeps = State.LOST_SWEEP, now, self.sweeps + 1
            self.plans.publish([], Mode.SWEEP, "lost")
            self._ev(now, f"LOST -> hover + 360 sweep #{self.sweeps}")

    def _sweep_logic(self, now: float) -> None:
        if now - self._sweep_t < 2 * math.pi / self.cfg.sweep_rate_rad_s:
            return
        if self.sweeps >= self.cfg.max_sweeps:
            return self._fail(now, "lost_after_sweeps")
        self.sweeps += 1
        self._sweep_t = now
        self.plans.publish([], Mode.SWEEP, "lost")
        self._ev(now, f"sweep #{self.sweeps}")

    # ---- arrival
    def _maybe_verify(self, now: float) -> None:
        if now < self._cooldown_until or self.hits < self.cfg.arrival_hits:
            return
        if self._dist_goal() >= self.cfg.arrival_dist_m:
            return
        f, _ = self.frames.get()
        if f is None:
            return
        self.plans.publish([], Mode.HOVER, "verifying")
        self.state, self._verify_t = State.VERIFY, now
        self._verify_depth = f.depth
        self._verify_id = self.vlm.submit(VLMKind.ARRIVAL, {"rgb": f.rgb, "depth": f.depth}, key="arrival")
        self._ev(now, f"verify requested (hits={self.hits}, d={self._dist_goal():.1f} m)")

    def _fallback_answer(self) -> str:
        return self.vlm.fallback_verify(self._verify_depth)

    def _apply_vlm(self, now: float) -> None:
        for r in self.vlm.poll():
            if r.kind is VLMKind.ARRIVAL and r.req_id == self._verify_id and self.state is State.VERIFY:
                self._verify_done(now, r.answer, f"{r.backend} {r.latency_s:.2f}s")
            elif r.kind is VLMKind.CAPTION and r.req_id in self._cap_ids:
                self.g.nodes[self._cap_ids.pop(r.req_id)]["caption"] = r.answer  # sole graph writer

    def _verify_done(self, now: float, answer: str, how: str) -> None:
        self.verify_log.append({"t": now, "answer": answer, "how": how})
        if str(answer).upper().startswith("YES"):
            self.state, self.reason = State.ARRIVED, "arrived"
            self.plans.publish([], Mode.HOVER, "arrived")
            self._ev(now, f"ARRIVED ({how})")
        else:
            self.hits, self._cooldown_until, self.state = 0, now + self.cfg.verify_cooldown_s, State.NAVIGATE
            self._publish_route(self.cur_node, "verify_no")
            self._ev(now, f"verify NO ({how}); resume")

    async def run(self, stop: asyncio.Event) -> None:
        self.start()
        period = 1.0 / self.cfg.tick_hz
        while not stop.is_set() and not self.done:
            self.tick()
            await asyncio.sleep(period)
        self.hb.beat()
