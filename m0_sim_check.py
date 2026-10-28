"""M0 sim check:  python m0_sim_check.py [--scene F] [--sim-config-path DIR] [--takeoff-alt 10] [--hover-s 5]
Connect -> load scene -> take off -> save 1 RGB + 1 depth (640x480) -> hover -> land -> report.
Exit codes: 0 pass, 1 gate failed, 2 preflight failed."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime

from vlnav.config import Config
from vlnav.frameio import save_frame
from vlnav.groundtruth import GroundTruth
from vlnav.preflight import check_camera_config
from vlnav.sim_interface import SimInterface

log = logging.getLogger("m0")


async def run(cfg: Config, out: str) -> dict:
    cam, fl = cfg.camera, cfg.flight
    sim = SimInterface(cfg)
    sim.connect()
    gt = GroundTruth(sim, fl.ground_z_world)
    rep: dict = {"checks": {}}
    if not gt.wait(cfg.sim.pose_wait_s):
        raise RuntimeError("no actual_pose within %.0f s" % cfg.sim.pose_wait_s)
    rep["spawn_world_ned"] = gt.spawn
    log.info("spawn world NED %s, yaw %.1f deg", gt.spawn, gt.yaw * 57.2958)
    landed = False
    try:
        await sim.takeoff()
        t0 = time.monotonic()
        while abs(gt.altitude - fl.takeoff_alt_m) > fl.alt_tol_m:
            vd = max(-3.0, min(3.0, 0.7 * ((gt.ground_z0 - fl.takeoff_alt_m) - gt.z)))
            await sim.velocity(0.0, 0.0, vd, 0.5)
            if time.monotonic() - t0 > fl.climb_timeout_s:
                raise RuntimeError("climb timeout, altitude %.2f" % gt.altitude)
        log.info("at altitude %.2f m", gt.altitude)
        rep["altitude_after_climb_m"] = round(gt.altitude, 2)

        rgb, depth = await asyncio.to_thread(sim.get_frame)
        sim._log("drone.get_images('%s', [ImageType.SCENE, ImageType.DEPTH_PERSPECTIVE])" % cam.camera_id)
        rep["saved"] = save_frame(out, "m0", rgb, depth)
        rep["checks"]["rgb_shape"] = list(rgb.shape)
        rep["checks"]["depth_shape"] = list(depth.shape)
        rep["checks"]["rgb_size_ok"] = (rgb.shape[1], rgb.shape[0]) == (cam.width, cam.height)
        rep["checks"]["depth_size_ok"] = (depth.shape[1], depth.shape[0]) == (cam.width, cam.height)
        rep["checks"]["depth_range_m"] = [float(import_nanmin(depth)), float(import_nanmax(depth))]
        rep["checks"]["fov_deg_config"], rep["checks"]["pitch_deg_config"] = cam.hfov_deg, cam.pitch_deg

        a0 = gt.altitude
        await sim.hover()
        await asyncio.sleep(fl.hover_s)
        rep["hover_alt_drift_m"] = round(gt.altitude - a0, 3)
        await sim.land(lambda: gt.altitude)
        landed = True
    finally:
        if not landed:
            try:
                await sim.land(lambda: gt.altitude)
            except Exception as exc:
                log.error("landing failed: %s", exc)
        sim.disconnect()
    rep["client_calls"] = sim.calls
    c = rep["checks"]
    rep["passed"] = bool(c.get("rgb_size_ok") and c.get("depth_size_ok")
                         and abs(rep["altitude_after_climb_m"] - fl.takeoff_alt_m) <= fl.alt_tol_m + 0.25)
    return rep


def import_nanmin(a):
    import numpy as np
    return np.nanmin(a)


def import_nanmax(a):
    import numpy as np
    return np.nanmax(a)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=None)
    ap.add_argument("--sim-config-path", default=None)
    ap.add_argument("--takeoff-alt", type=float, default=None)
    ap.add_argument("--hover-s", type=float, default=None)
    ap.add_argument("--swap-rgb", action="store_true")
    a = ap.parse_args()
    cfg = Config()
    if a.scene: cfg.sim.scene_config = a.scene
    if a.sim_config_path: cfg.sim.sim_config_path = a.sim_config_path
    if a.takeoff_alt is not None: cfg.flight.takeoff_alt_m = a.takeoff_alt
    if a.hover_s is not None: cfg.flight.hover_s = a.hover_s
    if a.swap_rgb:
        from dataclasses import replace
        cfg.camera = replace(cfg.camera, swap_rgb=True)
    out = os.path.join(cfg.runs_dir, f"m0_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out, exist_ok=True); os.makedirs(cfg.logs_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        handlers=[logging.FileHandler(os.path.join(out, "m0.log")), logging.StreamHandler(sys.stdout)])
    problems = check_camera_config(cfg, cfg.sim.sim_config_path or os.path.join(os.path.dirname(__file__), "sim_config"))
    if problems:
        for p in problems: log.error("preflight: %s", p)
        return 2
    log.info("preflight OK (camera %dx%d, FOV %.0f, pitch %.0f, GPS off)", cfg.camera.width, cfg.camera.height,
             cfg.camera.hfov_deg, cfg.camera.pitch_deg)
    try:
        rep = asyncio.run(run(cfg, out))
    except Exception as exc:   # SimConnectionError, climb timeout, no pose ...
        log.error("M0 FAILED: %s", exc)
        return 1
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
