"""Evaluation-only landmarks (P0, road centreline, X1) and the pure geometry on them. NED metres, yaw/heading in
degrees clockwise from north. No simulator or ground-truth-API import here, so both the survey tool and the
evaluator can use it."""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import yaml

NE = Tuple[float, float]


def wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def heading_deg(a: NE, b: NE) -> float:
    """Bearing of a->b, clockwise from north, [0, 360)."""
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 360.0


def polyline_length(pts: Sequence[NE]) -> float:
    return sum(math.hypot(q[0] - p[0], q[1] - p[1]) for p, q in zip(pts, pts[1:]))


def turn_left_deg(from_yaw_deg: float, to_heading_deg: float) -> float:
    """Left (counter-clockwise) turn needed to go from from_yaw to to_heading; negative = turn right."""
    return wrap180(from_yaw_deg - to_heading_deg)


def initial_road_heading(path: Sequence[NE], min_seg_m: float = 10.0) -> float:
    """Heading of the road's first stretch: from path[0] to the first point at least min_seg_m away."""
    for q in path[1:]:
        if math.hypot(q[0] - path[0][0], q[1] - path[0][1]) >= min_seg_m:
            return heading_deg(path[0], q)
    return heading_deg(path[0], path[-1])


def build_landmarks(p0_ned: Tuple[float, float, float], spawn_yaw_deg: float, centreline: List[NE], x1: NE,
                    max_alt_m: float = 0.0) -> Dict:
    if not centreline:
        raise ValueError("need at least one centreline point")
    route = [(p0_ned[0], p0_ned[1])] + list(centreline) + [x1]
    road = list(centreline) + [x1]
    hd = initial_road_heading(road)
    return {
        "p0": {"n": p0_ned[0], "e": p0_ned[1], "z": p0_ned[2], "yaw_deg": spawn_yaw_deg},
        "x1": {"n": x1[0], "e": x1[1]},
        "centreline": [[n, e] for n, e in centreline],
        "road_heading_deg": hd,
        "initial_turn_left_deg": turn_left_deg(spawn_yaw_deg, hd),
        "road_length_p0_to_x1_m": polyline_length(route),        # P0 -> centreline -> X1 (path length; SPL reference)
        "road_length_centreline_to_x1_m": polyline_length(road),
        "straight_line_p0_to_x1_m": math.hypot(x1[0] - p0_ned[0], x1[1] - p0_ned[1]),
        "max_altitude_flown_m": max_alt_m,
    }


def save_landmarks(path: str, lm: Dict) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(lm, f, sort_keys=False)


def load_landmarks(path: str) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)
