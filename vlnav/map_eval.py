"""Map evaluation against ground truth (evaluator side). Nodes carry a log-only 'gt' [n, e, yaw] expressed in the
same frame as odometry (origin = P0, NED axes)."""
from __future__ import annotations

import math
from typing import Dict, Sequence

import networkx as nx
import numpy as np


def _gt(a: dict, b: dict) -> float:
    return math.hypot(a["gt"][0] - b["gt"][0], a["gt"][1] - b["gt"][1])


def shortcut_stats(g: nx.Graph, nodes: Sequence[dict], tau: float) -> Dict[str, float]:
    """Precision = share of shortcut edges whose TRUE horizontal distance is <= tau (also reported at 2 tau)."""
    by_id = {n["id"]: n for n in nodes}
    d = [_gt(by_id[u], by_id[v]) for u, v, a in g.edges(data=True) if a["kind"] == "shortcut"]
    if not d:
        return {"n_shortcuts": 0, "precision_le_tau": float("nan"), "precision_le_2tau": float("nan"),
                "median_true_dist_m": float("nan"), "max_true_dist_m": float("nan")}
    d = np.array(d)
    return {"n_shortcuts": int(len(d)), "precision_le_tau": float((d <= tau).mean()),
            "precision_le_2tau": float((d <= 2 * tau).mean()), "median_true_dist_m": float(np.median(d)),
            "max_true_dist_m": float(d.max())}


def odometry_drift(nodes: Sequence[dict]) -> Dict[int, Dict[str, float]]:
    """Per root: horizontal odometry-vs-truth error at the last keyframe and the max over the pass."""
    out: Dict[int, Dict[str, float]] = {}
    for r in sorted({n["root"] for n in nodes}):
        err = [math.hypot(n["odo"][0] - n["gt"][0], n["odo"][1] - n["gt"][1]) for n in nodes if n["root"] == r]
        out[r] = {"final_err_m": err[-1], "max_err_m": max(err), "n": len(err)}
    return out
