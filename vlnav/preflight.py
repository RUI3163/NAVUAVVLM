"""Simulator-free preflight: the shipped jsonc files must match Config (camera size, FOV, pitch, GPS off)."""
from __future__ import annotations

import logging
import os
from typing import List

from .config import Config
from .jsonc import load_jsonc

log = logging.getLogger(__name__)
SCENE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sim_config")


def _floats(s: str) -> List[float]:
    return [float(x) for x in s.split()]


def check_camera_config(cfg: Config, sim_config_dir: str = SCENE_DIR) -> List[str]:
    """Returns a list of problems (empty = OK)."""
    cam, problems = cfg.camera, []
    scene = load_jsonc(os.path.join(sim_config_dir, cfg.sim.scene_config))
    robot_files = None
    for a in scene.get("actors", []):
        if a.get("name") == cfg.sim.vehicle_name:
            robot_files = a["robot-config"]
    if robot_files is None:
        return [f"vehicle {cfg.sim.vehicle_name} not in scene"]
    robot_files = [robot_files] if isinstance(robot_files, str) else robot_files
    sensors = []
    for rf in robot_files:
        sensors += load_jsonc(os.path.join(sim_config_dir, rf)).get("sensors", [])
    by_id = {s["id"]: s for s in sensors}
    s = by_id.get(cam.camera_id)
    if s is None:
        return [f"camera '{cam.camera_id}' not found (have {sorted(by_id)})"]
    if by_id.get("GPS", {}).get("enabled", False):
        problems.append("GPS sensor is enabled (GPS-denied project)")
    rpy = _floats(s["origin"]["rpy-deg"])
    if rpy != [cam.roll_deg, cam.pitch_deg, cam.yaw_deg]:
        problems.append(f"mount rpy-deg {rpy} != config {[cam.roll_deg, cam.pitch_deg, cam.yaw_deg]}")
    if tuple(_floats(s["origin"]["xyz"])) != tuple(cam.mount_xyz):
        problems.append(f"mount xyz {s['origin']['xyz']} != config {cam.mount_xyz}")
    caps = {c["image-type"]: c for c in s["capture-settings"] if c.get("capture-enabled")}
    if sorted(caps) != [0, 2]:
        problems.append(f"enabled image types {sorted(caps)} != [0 (scene), 2 (depth perspective)]")
    for t, c in caps.items():
        if (c["width"], c["height"]) != (cam.width, cam.height):
            problems.append(f"image-type {t} size {c['width']}x{c['height']} != {cam.width}x{cam.height}")
        if c["fov-degrees"] != cam.hfov_deg:
            problems.append(f"image-type {t} fov {c['fov-degrees']} != {cam.hfov_deg}")
    return problems
