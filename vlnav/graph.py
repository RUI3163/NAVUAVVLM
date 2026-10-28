"""Topological map graph (NetworkX). Nodes: dicts with id, root (flight), idx (order in root), odo [n,e,yaw].
Sequential edges join consecutive keyframes of one root. Shortcut edges join NON-adjacent keyframes whose odometry
distance <= tau, where tau = shortcut_factor x the MEASURED median sequential spacing (no hard-coded metres)."""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import networkx as nx
import numpy as np


def _d(a: dict, b: dict) -> float:
    return math.hypot(a["odo"][0] - b["odo"][0], a["odo"][1] - b["odo"][1])


def spacing_stats(nodes: Sequence[dict], min_move_m: float = 0.5) -> Dict[str, float]:
    """Distribution of consecutive-keyframe distances within each root (zero-move yaw keyframes excluded from the
    median so in-place turns do not drag it to 0)."""
    by_root: Dict[int, List[dict]] = {}
    for n in nodes:
        by_root.setdefault(n["root"], []).append(n)
    d = [_d(a, b) for ns in by_root.values() for a, b in zip(ns, ns[1:])]
    moving = [x for x in d if x >= min_move_m] or d
    q = lambda p: float(np.percentile(moving, p))
    return {"n_pairs": len(d), "n_moving": len(moving), "median": q(50), "p10": q(10), "p90": q(90),
            "min": float(min(d)), "max": float(max(d))}


def derive_tau(nodes: Sequence[dict], factor: float) -> float:
    return factor * spacing_stats(nodes)["median"]


def build_graph(nodes: Sequence[dict], tau: float, scope: str = "same_root") -> nx.Graph:
    if scope not in ("same_root", "any"):
        raise ValueError("scope must be 'same_root' or 'any'")
    g = nx.Graph()
    for n in nodes:
        g.add_node(n["id"], **{k: v for k, v in n.items() if k != "id"})
    for a, b in zip(nodes, nodes[1:]):
        if a["root"] == b["root"]:
            g.add_edge(a["id"], b["id"], kind="seq", weight=_d(a, b))
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            if scope == "same_root" and a["root"] != b["root"]:
                continue
            if a["root"] == b["root"] and b["idx"] - a["idx"] <= 1:
                continue                          # adjacent -> already a sequential edge
            d = _d(a, b)
            if d <= tau and not g.has_edge(a["id"], b["id"]):
                g.add_edge(a["id"], b["id"], kind="shortcut", weight=d)
    return g


def edge_counts(g: nx.Graph) -> Dict[str, int]:
    k = [d["kind"] for _, _, d in g.edges(data=True)]
    return {"nodes": g.number_of_nodes(), "edges": len(k), "seq": k.count("seq"), "shortcut": k.count("shortcut")}
