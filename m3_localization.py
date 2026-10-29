"""M3 localisation replay (offline, no simulator):
  python m3_localization.py --map-dir runs\\m2_<ts>\\map [--held-out 2] [--reembed cls|mean] [--device cpu]
  python m3_localization.py --all-folds [--vote-spread-factor 2.0]     # hold out each pass in turn, one summary row each
Map = every pass except the held-out one; the held-out pass is replayed frame by frame through the localiser.
Reports within-one-node / within-8-m, fused-position error (total and along/across the road), why frames got no fix,
and saves images for every failure."""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from datetime import datetime

import cv2
import numpy as np

from vlnav.config import Config
from vlnav.loc_eval import evaluate_replay
from vlnav.localization import Localizer
from vlnav.map_store import load_map

log = logging.getLogger("m3")


def _label(img, text):
    img = cv2.resize(img, (320, 240))
    cv2.rectangle(img, (0, 0), (320, 22), (0, 0, 0), -1)
    cv2.putText(img, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def failure_image(base, by_id, f, out):
    q, fx, tr = by_id[f["q_id"]], by_id[f["fix_node"]], by_id[f["true_node"]]
    ims = [cv2.imread(os.path.join(base, n["image"])) for n in (q, fx, tr)]
    if any(i is None for i in ims):
        return None
    row = np.hstack([_label(ims[0], f"query #{q['id']} (held-out)"),
                     _label(ims[1], f"FIX node {fx['id']} err {f['err_m']:.1f} m"),
                     _label(ims[2], f"true node {tr['id']}")])
    p = os.path.join(out, f"fail_q{q['id']:04d}.png"); cv2.imwrite(p, row); return p


def run_fold(cfg, nodes, emb, meta, base, mdir, held, out, pooling_label):
    mi = [i for i, n in enumerate(nodes) if n["root"] != held]
    qi = sorted([i for i, n in enumerate(nodes) if n["root"] == held], key=lambda i: nodes[i]["idx"])
    if not mi or not qi:
        print(f"held-out pass {held} leaves an empty map or query set")
        return None
    mnodes, qnodes = [nodes[i] for i in mi], [nodes[i] for i in qi]
    spacing = meta["spacing"]["median"]
    loc = Localizer(emb[mi], np.array([n["odo"][:2] for n in mnodes]), np.array([n["root"] for n in mnodes]),
                    np.array([n["idx"] for n in mnodes]), [n["id"] for n in mnodes], cfg.loc, spacing)
    results = [loc.step(emb[i], nodes[i]["odo"][:2]) for i in qi]
    rep = evaluate_replay(mnodes, qnodes, results, cfg.loc)
    os.makedirs(out, exist_ok=True)
    by_id = {n["id"]: n for n in nodes}
    fdir = os.path.join(out, "failures"); os.makedirs(fdir, exist_ok=True)
    rep["failure_images"] = [p for p in (failure_image(base, by_id, f, fdir) for f in rep["failures"][:20]) if p]
    rep["setup"] = {"map_dir": mdir, "held_out_pass": held, "map_nodes": len(mnodes), "query_frames": len(qnodes),
                    "spacing_m": spacing, "pooling": pooling_label, "loc_cfg": vars(cfg.loc)}
    json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=2)
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-dir", default=None)
    ap.add_argument("--held-out", type=int, default=None)
    ap.add_argument("--reembed", choices=["cls", "mean"], default=None, help="recompute embeddings from saved images")
    ap.add_argument("--device", default=None)
    ap.add_argument("--vote-spread-factor", type=float, default=None)
    ap.add_argument("--all-folds", action="store_true")
    a = ap.parse_args()
    cfg = Config()
    if a.vote_spread_factor is not None:
        cfg.loc.vote_spread_factor = a.vote_spread_factor
    mdir = a.map_dir or (sorted(glob.glob(os.path.join(cfg.runs_dir, "m2_*", "map")))[-1:] or [None])[0]
    if not mdir or not os.path.isdir(mdir):
        print("no map found: pass --map-dir runs\\m2_<ts>\\map"); return 2
    nodes, emb, g, meta = load_map(mdir)
    base = os.path.dirname(os.path.normpath(mdir))
    if a.reembed:
        from vlnav.features import Embedder
        cfg.features.pooling = a.reembed
        if a.device: cfg.features.device = a.device
        m = Embedder(cfg.features)
        emb = m.embed([cv2.imread(os.path.join(base, n["image"])) for n in nodes]); m.free()
    label = a.reembed or meta.get("pooling")
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    if a.all_folds:
        rows = []
        for held in sorted({n["root"] for n in nodes}):
            r = run_fold(cfg, nodes, emb, meta, base, mdir, held, os.path.join(cfg.runs_dir, f"m3_{stamp}_fold{held}"), label)
            if r is None:
                continue
            reasons = {}
            for f in r["no_fix_frames"]:
                reasons[f["reason"]] = reasons.get(f["reason"], 0) + 1
            rows.append({"held_out": held, "fix_rate": round(r["fix_rate"], 3), "one_node": round(r["within_one_node"], 3),
                         "within_8m": round(r["within_8m"], 3), "fused_med": round(r["fused_err_median_m"], 2),
                         "fused_max": round(r["fused_err_max_m"], 2), "fused_le5": round(r["fused_within_5m"], 3),
                         "north_med": round(r["fused_abs_north_median_m"], 2), "east_med": round(r["fused_abs_east_median_m"], 2),
                         "ungated_8m": round(r["ungated_top1_within_8m"], 3), "no_fix_reasons": reasons})
        print(json.dumps({"vote_spread_factor": cfg.loc.vote_spread_factor, "pooling": label, "folds": rows}, indent=2))
        return 0
    held = a.held_out if a.held_out is not None else meta.get("held_out_pass", cfg.map.held_out_pass)
    out = os.path.join(cfg.runs_dir, f"m3_{stamp}")
    rep = run_fold(cfg, nodes, emb, meta, base, mdir, held, out, label)
    if rep is None:
        return 2
    print(json.dumps({k: v for k, v in rep.items() if k not in ("frames", "failures")}, indent=2))
    print("report:", os.path.join(out, "report.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
