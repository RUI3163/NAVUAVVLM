"""A* on the NetworkX map graph. Edge weight = odometry distance (= Euclidean), so the straight-line heuristic is
admissible. Several equivalent goal nodes (same place seen in different passes) are allowed; the cheapest reachable
one wins. Nodes of different passes are different graph components unless cross-pass shortcuts exist."""
from __future__ import annotations

import math
from typing import Iterable, List, Optional, Tuple

import networkx as nx


def _xy(g: nx.Graph, n: int) -> Tuple[float, float]:
    o = g.nodes[n]["odo"]
    return float(o[0]), float(o[1])


def plan_path(g: nx.Graph, start: int, goals: Iterable[int]) -> Optional[Tuple[List[int], float]]:
    """-> (node path, cost m) to the cheapest reachable goal, or None."""
    best: Optional[Tuple[List[int], float]] = None
    for goal in goals:
        if goal not in g or start not in g or not nx.has_path(g, start, goal):
            continue
        gx, gy = _xy(g, goal)
        h = lambda a, b, gx=gx, gy=gy: math.hypot(_xy(g, a)[0] - gx, _xy(g, a)[1] - gy)
        path = nx.astar_path(g, start, goal, heuristic=h, weight="weight")
        cost = float(nx.path_weight(g, path, weight="weight"))
        if best is None or cost < best[1]:
            best = (path, cost)
    return best
