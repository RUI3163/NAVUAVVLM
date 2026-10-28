"""M2 mapping flights (Phase A):
  python m2_mapping.py --landmarks landmarks.yaml [--passes 3] [--device auto|cuda|cpu] [--pooling cls|mean]
Flies the passes, embeds the keyframes (after landing: never two heavy things on the GPU at once), builds the
graph, saves it to runs/m2_<ts>/map, reloads it and reports node/edge counts and shortcut precision.
Exit codes: 0 ok, 1 failed, 2 preflight failed."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import replace
from datetime import datetime

import cv2
import numpy as np

from vlnav.config import Config
from vlnav.graph import build_graph, derive_tau, edge_counts, spacing_stats
from vlnav.groundtruth import GroundTruth
from vlnav.landmarks import load_landmarks
from vlnav.map_eval import odometry_drift, shortcut_stats
from vlnav.map_store import load_map, save_map
from vlnav.mapping_flight import PassFlyer
from vlnav.preflight import check_camera_config
from vlnav.sim_interface import SimInterface

log = logging.getLogger("m2")


async def fly_all(cfg: Config, lm: dict, out: str):
    sim = SimInterface(cfg); sim.connect()
    gt = GroundTruth(sim, cfg.flight.ground_z_world)
    if not gt.wait(cfg.sim.pose_wait_s):
        raise RuntimeError("no actual_pose")
    nodes, yaw_sign = [], 1.0
    try:
        for k in range(cfg.map.n_passes):
            f = PassFlyer(sim, gt, cfg, lm, k, out)
            f.yaw_sign = yaw_sign
            nodes += await f.run()
            yaw_sign = f.yaw_sign
    finally:
        sim.disconnect()
    return nodes, sim.calls


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmarks", default="landmarks.yaml")
    ap.add_argument("--passes", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--pooling", choices=["cls", "mean"], default=None)
    ap.add_argument("--scope", choices=["same_root", "any"], default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--sim-config-path", default=None)
    a = ap.parse_args()
    cfg = Config()
    if a.passes: cfg.map.n_passes = a.passes
    if a.device: cfg.features.device = a.device
    if a.pooling: cfg.features.pooling = a.pooling
    if a.scope: cfg.map.shortcut_scope = a.scope
    if a.seed is not None: cfg.seed = a.seed
    if a.sim_config_path: cfg.sim.sim_config_path = a.sim_config_path
    out = os.path.join(cfg.runs_dir, f"m2_{datetime.now():%Y%m%d_%H%M%S}"); os.makedirs(out, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        handlers=[logging.FileHandler(os.path.join(out, "m2.log")), logging.StreamHandler(sys.stdout)])
    problems = check_camera_config(cfg, cfg.sim.sim_config_path or os.path.join(os.path.dirname(__file__), "sim_config"))
    if problems:
        for p in problems: log.error("preflight: %s", p)
        return 2
    if not os.path.isfile(a.landmarks):
        log.error("landmarks file not found: %s (copy the M1 landmarks.yaml here or pass --landmarks)", a.landmarks)
        return 2
    lm = load_landmarks(a.landmarks)
    try:
        nodes, calls = asyncio.run(fly_all(cfg, lm, out))
        for i, n in enumerate(nodes):
            n["id"] = i
        # ---- embeddings (after landing; sim client is idle) ----
        from vlnav.features import Embedder
        emb_model = Embedder(cfg.features)
        imgs = [cv2.imread(os.path.join(out, n["image"])) for n in nodes]
        emb = emb_model.embed(imgs); emb_model.free()
        # ---- graph ----
        sp = spacing_stats(nodes)
        tau = derive_tau(nodes, cfg.map.shortcut_factor)
        g = build_graph(nodes, tau, cfg.map.shortcut_scope)
        meta = {"tau_m": tau, "spacing": sp, "scope": cfg.map.shortcut_scope, "pooling": cfg.features.pooling,
                "held_out_pass": cfg.map.held_out_pass, "n_passes": cfg.map.n_passes, "seed": cfg.seed}
        mdir = os.path.join(out, "map"); save_map(mdir, nodes, emb, g, meta)
        n2, e2, g2, m2 = load_map(mdir)
        reload_ok = (n2 == nodes and np.array_equal(e2, emb.astype(np.float32)) and
                     sorted(map(sorted, g.edges())) == sorted(map(sorted, g2.edges())) and m2 == meta)
        rep = {"counts": edge_counts(g), "per_root_nodes": {r: sum(n["root"] == r for n in nodes) for r in range(cfg.map.n_passes)},
               "spacing_m": sp, "tau_m": tau, "shortcut": shortcut_stats(g, nodes, tau),
               "odometry_drift": odometry_drift(nodes), "embedding_shape": list(emb.shape),
               "embedding_norm_range": [float(np.linalg.norm(emb, axis=1).min()), float(np.linalg.norm(emb, axis=1).max())],
               "reload_identical": bool(reload_ok), "client_calls": calls, "map_dir": mdir}
    except Exception as exc:
        log.exception("M2 FAILED: %s", exc); return 1
    json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=2)
    print(json.dumps(rep, indent=2))
    return 0 if rep["reload_identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
