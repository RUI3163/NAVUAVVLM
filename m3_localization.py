"""M3 localisation replay (offline, no simulator):
  python m3_localization.py --map-dir runs\\m2_<ts>\\map [--held-out 2] [--reembed cls|mean] [--device cpu]
Map = every pass except the held-out one; the held-out pass is replayed frame by frame through the localiser.
Reports the share of fixes within one node and within 8 m, and saves images for every failure."""
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-dir", default=None)
    ap.add_argument("--held-out", type=int, default=None)
    ap.add_argument("--reembed", choices=["cls", "mean"], default=None, help="recompute embeddings from saved images")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    cfg = Config()
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
    held = a.held_out if a.held_out is not None else meta.get("held_out_pass", cfg.map.held_out_pass)
    mi = [i for i, n in enumerate(nodes) if n["root"] != held]
    qi = sorted([i for i, n in enumerate(nodes) if n["root"] == held], key=lambda i: nodes[i]["idx"])
    if not mi or not qi:
        print(f"held-out pass {held} leaves an empty map or query set"); return 2
    mnodes, qnodes = [nodes[i] for i in mi], [nodes[i] for i in qi]
    spacing = meta["spacing"]["median"]
    loc = Localizer(emb[mi], np.array([n["odo"][:2] for n in mnodes]), np.array([n["root"] for n in mnodes]),
                    np.array([n["idx"] for n in mnodes]), [n["id"] for n in mnodes], cfg.loc, spacing)
    results = [loc.step(emb[i], nodes[i]["odo"][:2]) for i in qi]
    rep = evaluate_replay(mnodes, qnodes, results, cfg.loc)
    out = os.path.join(cfg.runs_dir, f"m3_{datetime.now():%Y%m%d_%H%M%S}"); os.makedirs(out, exist_ok=True)
    by_id = {n["id"]: n for n in nodes}
    fdir = os.path.join(out, "failures"); os.makedirs(fdir, exist_ok=True)
    rep["failure_images"] = [p for p in (failure_image(base, by_id, f, fdir) for f in rep["failures"][:20]) if p]
    rep["setup"] = {"map_dir": mdir, "held_out_pass": held, "map_nodes": len(mnodes), "query_frames": len(qnodes),
                    "spacing_m": spacing, "pooling": a.reembed or meta.get("pooling"), "loc_cfg": vars(cfg.loc)}
    json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=2)
    summary = {k: v for k, v in rep.items() if k not in ("frames", "failures")}
    print(json.dumps(summary, indent=2)); print("report:", os.path.join(out, "report.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
