"""Node path -> waypoint list (Ramer-Douglas-Peucker simplification; start and end always kept)."""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import networkx as nx

XY = Tuple[float, float]


def _perp(p: XY, a: XY, b: XY) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dy)
    if L < 1e-9:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    return abs(dx * (a[1] - p[1]) - dy * (a[0] - p[0])) / L


def simplify(pts: Sequence[XY], tol_m: float) -> List[XY]:
    pts = list(pts)
    if len(pts) <= 2:
        return pts
    imax, dmax = 0, -1.0
    for i in range(1, len(pts) - 1):
        d = _perp(pts[i], pts[0], pts[-1])
        if d > dmax:
            imax, dmax = i, d
    if dmax <= tol_m:
        return [pts[0], pts[-1]]
    return simplify(pts[: imax + 1], tol_m)[:-1] + simplify(pts[imax:], tol_m)


def path_to_waypoints(g: nx.Graph, path: Sequence[int], tol_m: float) -> List[XY]:
    xy = [(float(g.nodes[n]["odo"][0]), float(g.nodes[n]["odo"][1])) for n in path]
    dedup = [p for i, p in enumerate(xy) if i == 0 or math.hypot(p[0] - xy[i - 1][0], p[1] - xy[i - 1][1]) > 1e-6]
    return simplify(dedup, tol_m)


def length(wps: Sequence[XY]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(wps, wps[1:]))
