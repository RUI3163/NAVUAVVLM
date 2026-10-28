"""M1 survey tool (EVALUATION ONLY: reads ground-truth pose). Keyboard-driven flight; marks the road centreline and
the first intersection X1 at the drone's current horizontal position -> landmarks.yaml.

Keys (press, the command lasts ~0.4 s):  w/s forward/back (body)  a/d strafe left/right  j/k yaw LEFT/RIGHT
  r/f climb/descend  SPACE hover  c mark centreline point  i mark intersection X1  g snapshot  u undo last mark
  x finish: write landmarks.yaml, land
Fly over the road CENTRE and press c about every 10-15 m going toward the intersection; hover over the centre of
the intersection and press i."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import queue
import sys
import threading
import time

from .config import Config
from .frameio import save_frame
from .groundtruth import GroundTruth
from .landmarks import build_landmarks, save_landmarks
from .sim_interface import SimInterface

log = logging.getLogger("survey")
STEP_V, STEP_YAW_DPS, STEP_VZ, CMD_S = 3.0, 30.0, 1.5, 0.4


def _key_thread(q: "queue.Queue[str]", stop: threading.Event) -> None:
    if os.name == "nt":
        import msvcrt
        while not stop.is_set():
            if msvcrt.kbhit():
                q.put(msvcrt.getwch().lower())
            else:
                time.sleep(0.02)
    else:                                   # fallback: type keys then Enter
        for line in sys.stdin:
            for ch in line.rstrip("\n") or " ":
                q.put(ch.lower())
            if stop.is_set():
                break


async def run(cfg: Config, out_dir: str, yaw_sign: float, alt_m: float) -> dict:
    sim = SimInterface(cfg); sim.connect()
    gt = GroundTruth(sim, cfg.flight.ground_z_world)
    if not gt.wait(cfg.sim.pose_wait_s):
        raise RuntimeError("no actual_pose")
    p0, yaw0 = gt.spawn, math.degrees(gt.yaw) % 360.0
    log.info("P0 (spawn) NED %s, yaw %.1f deg", p0, yaw0)
    keys: "queue.Queue[str]" = queue.Queue(); stop = threading.Event()
    threading.Thread(target=_key_thread, args=(keys, stop), daemon=True).start()
    centre, x1, max_alt, target_alt = [], None, 0.0, alt_m
    trail = open(os.path.join(out_dir, "trail.jsonl"), "w")
    last_log = 0.0
    try:
        await sim.takeoff()
        print(__doc__)
        k, t_cmd, cmd = None, 0.0, (0.0, 0.0, 0.0)
        while True:
            while not keys.empty():
                k = keys.get()
                yaw = gt.yaw
                fwd = {"w": 1, "s": -1}.get(k, 0) * STEP_V
                right = {"d": 1, "a": -1}.get(k, 0) * STEP_V
                yr = {"k": 1, "j": -1}.get(k, 0) * math.radians(STEP_YAW_DPS) * yaw_sign   # j = yaw LEFT
                target_alt += {"r": 1, "f": -1}.get(k, 0) * 1.0
                if k in "wsadjk":
                    cmd = (fwd * math.cos(yaw) - right * math.sin(yaw), fwd * math.sin(yaw) + right * math.cos(yaw), yr)
                    t_cmd = time.monotonic()
                elif k == " ":
                    cmd = (0.0, 0.0, 0.0); t_cmd = 0.0
                elif k == "c":
                    centre.append((gt.n, gt.e)); print(f"centreline #{len(centre)}: N {gt.n:.1f} E {gt.e:.1f}")
                elif k == "i":
                    x1 = (gt.n, gt.e); print(f"X1 marked: N {gt.n:.1f} E {gt.e:.1f}")
                elif k == "u":
                    if x1 is not None: x1 = None; print("X1 cleared")
                    elif centre: centre.pop(); print("last centreline point removed")
                elif k == "g":
                    rgb, dep = await asyncio.to_thread(sim.get_frame)
                    save_frame(out_dir, f"snap_{int(time.time())}", rgb, dep); print("snapshot saved")
                elif k == "x":
                    raise StopIteration
            active = time.monotonic() - t_cmd < CMD_S
            vn, ve, yr = cmd if active else (0.0, 0.0, 0.0)
            vd = max(-2.0, min(2.0, 0.7 * ((gt.ground_z0 - target_alt) - gt.z)))
            await sim.velocity(vn, ve, vd, 0.2, yaw_rate=yr)
            max_alt = max(max_alt, gt.altitude)
            if time.monotonic() - last_log > 1.0:
                last_log = time.monotonic()
                trail.write(json.dumps({"t": last_log, "n": gt.n, "e": gt.e, "alt": gt.altitude,
                                        "yaw_deg": math.degrees(gt.yaw) % 360.0}) + "\n"); trail.flush()
    except StopIteration:
        pass
    finally:
        stop.set(); trail.close()
        try:
            await sim.hover(); await sim.land(lambda: gt.altitude)
        except Exception as exc:
            log.error("landing failed: %s", exc)
        sim.disconnect()
    if x1 is None or not centre:
        raise RuntimeError("need >=1 centreline point (c) and the intersection (i); nothing saved")
    lm = build_landmarks(p0, yaw0, centre, x1, max_alt)
    save_landmarks(os.path.join(out_dir, "landmarks.yaml"), lm)
    lm["client_calls"] = sim.calls
    return lm
