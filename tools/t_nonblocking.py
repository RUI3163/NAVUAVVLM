"""Tiny sim test for the M4 control loop (run from the repo root, simulator running):
    python tools\\t_nonblocking.py --landmarks landmarks.yaml
Answers three questions with measurements (ground truth is used HERE only to measure):
  A  Does a velocity command EXPIRE by itself after `duration` (drone stops), or does it persist?
  B  Can a 20 Hz loop send fire-and-forget commands (no await of the completion task)? loop period / submit latency.
  C  Is the 20 Hz loop undisturbed while get_frame() runs in a worker thread at ~3 Hz?
Flies the surveyed road at 10 m (no obstacles there in the survey), then returns to P0 and lands."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vlnav.config import Config                      # noqa: E402
from vlnav.groundtruth import GroundTruth            # noqa: E402
from vlnav.landmarks import load_landmarks           # noqa: E402
from vlnav.sim_interface import SimInterface         # noqa: E402

log = logging.getLogger("t_nonblocking")


def stats(x):
    a = np.array(x, float) * 1000.0
    if not len(a):
        return {"n": 0}
    return {"n": int(len(a)), "mean_ms": round(float(a.mean()), 2), "p50_ms": round(float(np.percentile(a, 50)), 2),
            "p95_ms": round(float(np.percentile(a, 95)), 2), "max_ms": round(float(a.max()), 2)}


def vd(gt, alt):
    return max(-2.0, min(2.0, 0.7 * ((gt.ground_z0 - alt) - gt.z)))


async def goto(sim, gt, alt, n, e, timeout=90.0):
    t0 = time.monotonic()
    while True:
        dn, de = n - gt.n, e - gt.e
        d = math.hypot(dn, de)
        if d < 1.0:
            return
        sp = min(3.0, max(0.8, 0.6 * d))
        await sim.velocity(dn / d * sp, de / d * sp, vd(gt, alt), 0.2)
        if time.monotonic() - t0 > timeout:
            raise RuntimeError(f"goto timeout, {d:.1f} m left")


async def test_expiry(sim, gt, alt):
    n0 = gt.n; a0 = gt.altitude
    await sim.velocity_nowait(3.0, 0.0, vd(gt, alt), 0.15)
    t0, samples, min_alt = time.monotonic(), [], gt.altitude
    while time.monotonic() - t0 < 3.0:
        samples.append((time.monotonic() - t0, gt.n - n0)); min_alt = min(min_alt, gt.altitude)
        await asyncio.sleep(0.05)
    at = lambda t: min(samples, key=lambda s: abs(s[0] - t))[1]
    last = [s for s in samples if s[0] >= 2.0]
    speed_end = (last[-1][1] - last[0][1]) / max(last[-1][0] - last[0][0], 1e-6)
    return {"disp_m_at_0.5s": round(at(0.5), 2), "disp_m_at_1s": round(at(1.0), 2), "disp_m_at_3s": round(at(3.0), 2),
            "speed_mps_last_second": round(speed_end, 3), "min_altitude_drop_m": round(a0 - min_alt, 2),
            "verdict": "expires (drone stops)" if abs(speed_end) < 0.3 else "PERSISTS (drone keeps moving)"}


async def stream(sim, gt, alt, e_road, secs, speed, with_images):
    period = 0.05
    periods, submit, pend, frame_lat, img_tasks, track = [], [], [], [], [], []
    t_start = time.monotonic(); last = t_start; nxt = t_start; next_img = t_start
    while time.monotonic() - t_start < secs:
        now = time.monotonic(); periods.append(now - last); last = now
        t = time.monotonic()
        await sim.velocity_nowait(speed, max(-1.0, min(1.0, e_road - gt.e)), vd(gt, alt), 0.15)
        submit.append(time.monotonic() - t); pend.append(sim.pending_commands); track.append((now, gt.n))
        if with_images and now >= next_img and all(x.done() for x in img_tasks):
            next_img = now + 0.33; t_img = time.monotonic()
            task = asyncio.create_task(asyncio.to_thread(sim.get_frame, False))
            task.add_done_callback(lambda f, t0=t_img: frame_lat.append(time.monotonic() - t0))
            img_tasks.append(task)
        nxt += period
        if nxt < time.monotonic() - period:
            nxt = time.monotonic()
        await asyncio.sleep(max(0.0, nxt - time.monotonic()))
    for t in img_tasks:
        await t                                       # surface any exception from the worker thread
    half = track[len(track) // 2:]
    v = (half[-1][1] - half[0][1]) / max(half[-1][0] - half[0][0], 1e-6)
    p = periods[1:]
    return {"commanded_mps": speed, "achieved_mps_2nd_half": round(v, 2), "loop_hz": round(1.0 / float(np.mean(p)), 2),
            "max_command_gap_ms": round(max(p) * 1000, 1), "loop_period": stats(p), "submit_latency": stats(submit),
            "max_pending_tasks": int(max(pend)), "frames": len(frame_lat), "frame_latency": stats(frame_lat)}


async def run(cfg, lm, out):
    sim = SimInterface(cfg); sim.connect()
    gt = GroundTruth(sim, cfg.flight.ground_z_world)
    if not gt.wait(cfg.sim.pose_wait_s):
        raise RuntimeError("no actual_pose")
    alt, p0 = cfg.flight.takeoff_alt_m, (lm["p0"]["n"], lm["p0"]["e"])
    cl = lm["centreline"]; e_road = sum(p[1] for p in cl) / len(cl); n_south = cl[0][0]
    rep = {}
    try:
        await sim.takeoff()
        t0 = time.monotonic()
        while abs(gt.altitude - alt) > cfg.flight.alt_tol_m:
            await sim.velocity(0.0, 0.0, vd(gt, alt), 0.5)
            if time.monotonic() - t0 > cfg.flight.climb_timeout_s:
                raise RuntimeError("climb timeout")
        await goto(sim, gt, alt, n_south, e_road)
        await sim.hover(); await asyncio.sleep(2.0)
        rep["A_expiry"] = await test_expiry(sim, gt, alt); log.info("A %s", rep["A_expiry"])
        await sim.hover(); await asyncio.sleep(1.0)
        rep["B_stream_no_images"] = await stream(sim, gt, alt, e_road, 6.0, 2.0, False); log.info("B %s", rep["B_stream_no_images"])
        await sim.hover(); await asyncio.sleep(2.0)
        rep["C_stream_with_images"] = await stream(sim, gt, alt, e_road, 6.0, 2.0, True); log.info("C %s", rep["C_stream_with_images"])
        await sim.hover(); await asyncio.sleep(1.0)
        await goto(sim, gt, alt, p0[0], p0[1])
        await sim.hover()
        await sim.land(lambda: gt.altitude)
    except Exception:
        try:
            await sim.hover(); await sim.land(lambda: gt.altitude)
        except Exception as exc:
            log.error("landing failed: %s", exc)
        raise
    finally:
        sim.disconnect()
    b, c = rep["B_stream_no_images"], rep["C_stream_with_images"]
    rep["summary"] = {"B_hz": b["loop_hz"], "C_hz": c["loop_hz"], "C_max_gap_ms": c["max_command_gap_ms"],
                      "images_disturb_loop": bool(c["max_command_gap_ms"] > 2.0 * max(b["max_command_gap_ms"], 60.0))}
    rep["client_calls"] = sim.calls
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--landmarks", default="landmarks.yaml")
    ap.add_argument("--sim-config-path", default=None); a = ap.parse_args()
    cfg = Config()
    if a.sim_config_path: cfg.sim.sim_config_path = a.sim_config_path
    out = os.path.join(cfg.runs_dir, f"tnb_{datetime.now():%Y%m%d_%H%M%S}"); os.makedirs(out, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        handlers=[logging.FileHandler(os.path.join(out, "t.log")), logging.StreamHandler(sys.stdout)])
    if not os.path.isfile(a.landmarks):
        log.error("landmarks file not found: %s", a.landmarks); return 2
    try:
        rep = asyncio.run(run(cfg, load_landmarks(a.landmarks), out))
    except Exception as exc:
        log.exception("FAILED: %s", exc); return 1
    json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=2); print(json.dumps(rep, indent=2)); return 0


if __name__ == "__main__":
    sys.exit(main())
