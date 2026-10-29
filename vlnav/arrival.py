"""Arrival logic (pure). Visual part of the check: localiser reports the goal node (or a direct graph neighbour) for
`consec_frames` frames in a row AND the corrected odometry is within `pos_tol_m` of the goal -> ask the VLM
("Is the drone above or at a road intersection?"); STOP only after a yes."""
from __future__ import annotations

import math
from typing import Iterable, Optional, Set, Tuple

import networkx as nx

from .config import ArrivalCfg

XY = Tuple[float, float]


def accept_set(g: nx.Graph, goal_nodes: Iterable[int]) -> Set[int]:
    s: Set[int] = set()
    for n in goal_nodes:
        s.add(n)
        s.update(g.neighbors(n))
    return s


class ArrivalChecker:
    def __init__(self, accept_nodes: Set[int], goal_xy: XY, cfg: ArrivalCfg):
        self.accept, self.goal_xy, self.cfg = set(accept_nodes), goal_xy, cfg
        self.streak = 0
        self.vetoes = 0
        self.arrived = False

    def update(self, fix_node: Optional[int], pos_xy: XY) -> str:
        """-> 'idle' | 'building' | 'ask_vlm' | 'arrived'. Call once per perception frame."""
        if self.arrived:
            return "arrived"
        near = math.hypot(pos_xy[0] - self.goal_xy[0], pos_xy[1] - self.goal_xy[1]) <= self.cfg.pos_tol_m
        if fix_node is not None and fix_node in self.accept and near:
            self.streak += 1
        else:
            self.streak = 0
        if self.streak >= self.cfg.consec_frames:
            return "ask_vlm"
        return "building" if self.streak else "idle"

    def vlm_answer(self, yes: bool) -> str:
        if yes:
            self.arrived = True
            return "arrived"
        self.vetoes += 1
        self.streak = 0                                    # require a fresh streak before asking again
        return "idle"
