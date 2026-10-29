"""Replay evaluation of the localiser against ground truth (evaluator side only).
within-one-node: the fix's node is the ground-truth-nearest node of ITS OWN pass (same travel heading, within
`same_heading_deg`) or that node's sequential neighbour (|idx diff| <= 1).
within-8-m: ground-truth distance between the fix node and the query <= match_radius_m (any leg)."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import LocCfg
from .localization import LocResult


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def evaluate_replay(map_nodes: Sequence[dict], query_nodes: Sequence[dict], results: Sequence[LocResult],
                    cfg: LocCfg) -> Dict:
    """map_nodes/query_nodes: dicts with id, root, idx, gt [n, e, yaw]. results[i] belongs to query_nodes[i];
    Fix.index / LocResult.*top1 are rows into map_nodes."""
    mgt = np.array([n["gt"][:2] for n in map_nodes]); myaw = np.array([n["gt"][2] for n in map_nodes])
    mroot = np.array([n["root"] for n in map_nodes]); midx = np.array([n["idx"] for n in map_nodes])
    hd = math.radians(cfg.same_heading_deg)
    frames: List[dict] = []
    fails: List[dict] = []
    raw_ok = gated_ok = n_gated = 0
    for qi, (q, r) in enumerate(zip(query_nodes, results)):
        qxy, qyaw = np.array(q["gt"][:2]), q["gt"][2]
        dall = np.linalg.norm(mgt - qxy, axis=1)
        raw_ok += int(dall[r.raw_top1] <= cfg.match_radius_m)
        if r.top1 is not None:
            n_gated += 1
            gated_ok += int(dall[r.top1] <= cfg.match_radius_m)
        row = {"q_id": q["id"], "q_idx": q["idx"], "fix_node": None, "reason": r.reason, "n_gated": r.n_gated,
               "gate_m": round(r.gate_m, 2), "sigma_m": round(r.sigma_m, 2),
               "spread_m": None if r.spread_m is None else round(r.spread_m, 2),
               "top1_node": None if r.top1 is None else map_nodes[r.top1]["id"],
               "top1_err_m": None if r.top1 is None else round(float(dall[r.top1]), 2)}
        if r.fix is not None:
            f = r.fix.index
            same = np.where((mroot == mroot[f]) & (np.abs([_wrap(y - qyaw) for y in myaw]) <= hd))[0]
            pool = same if len(same) else np.where(mroot == mroot[f])[0]
            near = pool[dall[pool] <= dall[pool].min() + cfg.colocate_m]       # co-located candidates (e.g. turn keyframes)
            tstar = int(near[np.argmin([abs(_wrap(myaw[i] - qyaw)) for i in near])])   # ties -> closest heading
            err = float(dall[f])
            co = bool(abs(_wrap(myaw[f] - myaw[tstar])) <= math.radians(cfg.colocate_heading_deg) and
                      np.linalg.norm(mgt[f] - mgt[tstar]) <= cfg.colocate_m)    # in-place yaw keyframes = one place
            one = bool(abs(int(midx[f]) - int(midx[tstar])) <= 1 or co)
            dv = np.array(r.fix.xy) - qxy
            fused = float(np.linalg.norm(dv))                                    # position the controller would use
            row.update({"fix_node": map_nodes[f]["id"], "true_node": map_nodes[tstar]["id"], "err_m": err,
                        "fused_err_m": fused, "fused_err_n_m": float(dv[0]), "fused_err_e_m": float(dv[1]), "within_one_node": one, "within_8m": bool(err <= cfg.match_radius_m)})
            if not (one and err <= cfg.match_radius_m):
                fails.append({"q_id": q["id"], "fix_node": map_nodes[f]["id"], "true_node": map_nodes[tstar]["id"],
                              "err_m": err, "within_one_node": one, "within_8m": bool(err <= cfg.match_radius_m)})
        frames.append(row)
    fixed = [f for f in frames if f["fix_node"] is not None]
    nq = max(len(frames), 1)
    errs = np.array([f["err_m"] for f in fixed]) if fixed else np.array([])
    ferr = np.array([f["fused_err_m"] for f in fixed]) if fixed else np.array([])
    fn = np.array([f["fused_err_n_m"] for f in fixed]) if fixed else np.array([])
    fe = np.array([f["fused_err_e_m"] for f in fixed]) if fixed else np.array([])
    nofix = [{k: f[k] for k in ("q_idx", "reason", "n_gated", "gate_m", "sigma_m", "spread_m", "top1_node", "top1_err_m")}
             for f in frames if f["fix_node"] is None]
    return {
        "n_frames": len(frames), "n_fixes": len(fixed), "fix_rate": len(fixed) / nq,
        "first_fix_frame": next((i for i, f in enumerate(frames) if f["fix_node"] is not None), None),
        "within_one_node": float(np.mean([f["within_one_node"] for f in fixed])) if fixed else float("nan"),
        "within_8m": float(np.mean([f["within_8m"] for f in fixed])) if fixed else float("nan"),
        "right_place_wrong_node": int(sum(1 for f in fixed if f["within_8m"] and not f["within_one_node"])),
        "err_median_m": float(np.median(errs)) if len(errs) else float("nan"),
        "err_mean_m": float(errs.mean()) if len(errs) else float("nan"),
        "err_max_m": float(errs.max()) if len(errs) else float("nan"),
        "fused_err_median_m": float(np.median(ferr)) if len(ferr) else float("nan"),
        "fused_err_mean_m": float(ferr.mean()) if len(ferr) else float("nan"),
        "fused_err_max_m": float(ferr.max()) if len(ferr) else float("nan"),
        "fused_within_5m": float((ferr <= 5.0).mean()) if len(ferr) else float("nan"),
        "fused_abs_north_median_m": float(np.median(np.abs(fn))) if len(fn) else float("nan"),   # along the road (runs north)
        "fused_abs_east_median_m": float(np.median(np.abs(fe))) if len(fe) else float("nan"),    # across the road
        "fused_abs_north_max_m": float(np.abs(fn).max()) if len(fn) else float("nan"),
        "fused_abs_east_max_m": float(np.abs(fe).max()) if len(fe) else float("nan"),
        "no_fix_frames": nofix,
        "ungated_top1_within_8m": raw_ok / nq,
        "gated_top1_within_8m": gated_ok / max(n_gated, 1),
        "n_failures": len(fails), "failures": fails, "frames": frames,
    }
