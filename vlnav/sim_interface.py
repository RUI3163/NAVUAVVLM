"""Thin Project AirSim wrapper for NAVIGATION-side code: connect, takeoff, velocity, hover, land, images.
It deliberately exposes NO pose. Ground-truth pose lives in groundtruth.py (evaluator/logger only).
Call patterns come from the reference repo (TFCUNAVUAV/vln/sim_client.py); get_images / ImageType from
ProjectAirSim client/python/projectairsim/src/projectairsim/{robot.py,types.py}; decode from utils.unpack_image."""
from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import Config
from .frameio import depth_from_msg, rgb_from_msg
from .preflight import SCENE_DIR

log = logging.getLogger(__name__)


class SimConnectionError(RuntimeError):
    pass


class SimInterface:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = self.world = self.drone = None
        self.calls: List[str] = []              # exact client calls made, for the M0 report

    def _log(self, s: str) -> None:
        self.calls.append(s)
        log.info("client call: %s", s)

    def connect(self) -> None:
        try:
            from projectairsim import Drone, ProjectAirSimClient, World
        except ImportError as exc:
            raise SimConnectionError("pip install -e ProjectAirSim/client/python/projectairsim") from exc
        s = self.cfg.sim
        cfg_dir = s.sim_config_path or SCENE_DIR
        if not os.path.isfile(os.path.join(cfg_dir, s.scene_config)):
            raise SimConnectionError(f"{s.scene_config} not found in {cfg_dir}")
        try:
            self.client = ProjectAirSimClient()
            self._log("ProjectAirSimClient()"); self.client.connect(); self._log("client.connect()")
            self.world = World(self.client, s.scene_config, delay_after_load_sec=s.delay_after_load_sec,
                               sim_config_path=cfg_dir)
            self._log(f"World(client, '{s.scene_config}', delay_after_load_sec={s.delay_after_load_sec}, "
                      f"sim_config_path='{cfg_dir}')")
            self.drone = Drone(self.client, self.world, s.vehicle_name)
            self._log(f"Drone(client, world, '{s.vehicle_name}')")
        except Exception as exc:
            raise SimConnectionError(f"Could not connect to Project AirSim: {exc}") from exc

    def disconnect(self) -> None:
        if self.client is not None:
            try:
                self.client.disconnect(); self._log("client.disconnect()")
            except Exception:
                pass

    # ---- control (asyncio; every *_async returns a task that is awaited) ----
    async def takeoff(self) -> None:
        self.drone.enable_api_control(); self._log("drone.enable_api_control()")
        self.drone.arm(); self._log("drone.arm()")
        await (await self.drone.takeoff_async()); self._log("await (await drone.takeoff_async())")

    async def velocity(self, vn: float, ve: float, vd: float, duration: float, yaw_rate: float = 0.0) -> None:
        """World-NED velocity (m/s) + optional yaw RATE (rad/s, sign convention checked by the survey tool).
        Source: projectairsim drone.py move_by_velocity_async(v_north, v_east, v_down, duration, yaw_control_mode,
        yaw_is_rate=True, yaw)."""
        if not any(c.startswith("drone.move_by_velocity_async(vn") for c in self.calls):
            self._log("drone.move_by_velocity_async(vn, ve, vd, duration=duration, yaw_is_rate=True, yaw=yaw_rate)")
        await (await self.drone.move_by_velocity_async(vn, ve, vd, duration=duration, yaw_is_rate=True, yaw=yaw_rate))

    async def hover(self) -> None:
        await (await self.drone.hover_async()); self._log("await (await drone.hover_async())")

    async def land(self, altitude_fn=None, handoff_m: float = 3.0) -> None:
        """Fast velocity descent to handoff_m (land_async alone is very slow), then land_async."""
        import time
        t0 = time.monotonic()
        while altitude_fn is not None and altitude_fn() > handoff_m + 0.5 and time.monotonic() - t0 < 30:
            await (await self.drone.move_by_velocity_async(0.0, 0.0, 3.0, duration=0.5))
        self._log("drone.move_by_velocity_async(0, 0, 3.0, duration=0.5) repeated to handoff height")
        await (await self.drone.land_async()); self._log("await (await drone.land_async())")
        self.drone.disarm(); self._log("drone.disarm()")
        self.drone.disable_api_control(); self._log("drone.disable_api_control()")

    # ---- images (blocking request; call via asyncio.to_thread from the control loop) ----
    def get_frame(self, want_depth: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """RGB (BGR uint8) + optionally perspective depth (float32 m, NaN = no hit) from a single request."""
        from projectairsim.types import ImageType
        cam = self.cfg.camera
        types = [ImageType.SCENE] + ([ImageType.DEPTH_PERSPECTIVE] if want_depth else [])
        imgs: Dict[int, dict] = self.drone.get_images(cam.camera_id, types)
        rgb = rgb_from_msg(imgs[ImageType.SCENE], swap_rgb=cam.swap_rgb)
        depth = depth_from_msg(imgs[ImageType.DEPTH_PERSPECTIVE], max_m=cam.depth_max_m) if want_depth else None
        return rgb, depth

    # ---- IMU gyro (yaw rate) for odometry; pattern from the reference repo's _on_imu ----
    def subscribe_gyro(self, cb) -> None:
        """cb(wz_rad_s, t_monotonic) is called on the client's receive thread; it must only copy and return."""
        topic = self.drone.sensors[self.cfg.sim.imu_id]["imu_kinematics"]
        self.client.subscribe(topic, lambda _t, m: cb(float(m["angular_velocity"]["z"]), time.monotonic()))
        self._log(f'client.subscribe(drone.sensors["{self.cfg.sim.imu_id}"]["imu_kinematics"], cb)')
