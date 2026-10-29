"""Wiring: SimIO (the only place that awaits the simulator) + mission runner."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import networkx as nx
import numpy as np

from vlnav.controller import ControllerConfig, ControlLoop, VelocityCmd
from vlnav.mission import MissionConfig, MissionManager
from vlnav.odometry import DeadReckoning, ImuSample, OdomNoise
from vlnav.perception import Frame, Localizer, PerceptionConfig, PerceptionWorker
from vlnav.plan_buffer import Heartbeat, LatestSlot, PlanBuffer
from vlnav.vlm_worker import VLMClient, VLMConfig
from vlnav.graph_utils import node_xy


class SimAdapter(Protocol):
    async def imu(self) -> ImuSample: ...
    async def frame(self) -> Optional[Frame]: ...
    async def send_velocity(self, cmd: VelocityCmd) -> None: ...


@dataclass
class M4Config:
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    mission: MissionConfig = field(default_factory=MissionConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    odom_noise: OdomNoise = field(default_factory=OdomNoise)
    imu_hz: float = 50.0
    frame_hz: float = 10.0
    cmd_resend_hz: float = 20.0


class SimIO:
    """Motion task (IMU->odometry, cmd->sim) and frame task. Slow sim calls here
    can never stall the control loop: it only touches slots."""

    def __init__(self, sim: SimAdapter, odom: DeadReckoning, frames: LatestSlot, cmds: LatestSlot, cfg: M4Config):
        self.sim, self.odom, self.frames, self.cmds, self.cfg = sim, odom, frames, cmds, cfg

    async def run_motion(self, stop: asyncio.Event) -> None:
        last = time.monotonic()
        period = 1.0 / self.cfg.imu_hz
        while not stop.is_set():
            imu = await self.sim.imu()
            now = time.monotonic()
            self.odom.step(imu, now - last)
            last = now
            cmd, _ = self.cmds.get()
            if cmd is not None:
                await self.sim.send_velocity(cmd)
            await asyncio.sleep(period)

    async def run_frames(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            f = await self.sim.frame()
            if f is not None:
                self.frames.put(f)
            await asyncio.sleep(1.0 / self.cfg.frame_hz)


@dataclass
class MissionResult:
    success: bool
    state: str
    reason: str
    duration_s: float
    final_odom_xy: tuple[float, float]
    control_hz: float
    hover_share: float
    vlm_latency_mean_s: float
    perception_latency_mean_s: float
    events: list = field(default_factory=list)
    verify_log: list = field(default_factory=list)


async def run_mission(sim: SimAdapter, g: nx.Graph, start_node: Any, goal_node: Any, localizer: Localizer,
                      cfg: M4Config, start_yaw: float = 0.0, timeout_s: float = 300.0) -> MissionResult:
    x0, y0 = node_xy(g, start_node)
    odom = DeadReckoning(cfg.odom_noise, x0, y0, start_yaw)
    plans, hb = PlanBuffer(), Heartbeat()
    frames, fixes, cmds = LatestSlot(), LatestSlot(), LatestSlot()
    vlm = VLMClient(cfg.vlm)
    vlm.start()
    cfg.mission.mission_timeout_s = timeout_s
    mm = MissionManager(g, start_node, goal_node, plans, hb, fixes, frames, odom, vlm, cfg.mission)
    ctl = ControlLoop(cfg.controller, plans, odom, frames, cmds, hb)
    per = PerceptionWorker(cfg.perception, localizer, frames, odom, fixes)
    io = SimIO(sim, odom, frames, cmds, cfg)
    stop = asyncio.Event()
    t0 = time.monotonic()
    mm.start()
    tasks = [asyncio.create_task(c) for c in (
        io.run_motion(stop), io.run_frames(stop), ctl.run(stop), per.run(stop), mm.run(stop))]
    try:
        while not mm.done and time.monotonic() - t0 < timeout_s + 2:
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.5)  # let hover command reach the sim
    finally:
        stop.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        vlm.stop()
    o = odom.state()
    dts = ctl.tick_dts[1:] or [1.0]
    mean = lambda a: float(np.mean(a)) if a else 0.0
    return MissionResult(mm.success, mm.state.value, mm.reason, time.monotonic() - t0, (o.x, o.y),
                         1.0 / float(np.mean(dts)), ctl.hover_ticks / max(1, ctl.ticks),
                         mean(vlm.latencies), mean(per.latencies), mm.events, mm.verify_log)
