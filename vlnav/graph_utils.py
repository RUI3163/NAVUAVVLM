from __future__ import annotations

import math
from typing import Any

import networkx as nx


def node_xy(g: nx.Graph, n: Any) -> tuple[float, float]:
    """Node position in the map frame. Adapt here if M2 stores it differently."""
    d = g.nodes[n]
    if "xy" in d:
        return float(d["xy"][0]), float(d["xy"][1])
    if "pos" in d:
        return float(d["pos"][0]), float(d["pos"][1])
    return float(d["x"]), float(d["y"])


def edge_cost(g: nx.Graph, u: Any, v: Any, d: dict) -> float:
    if d.get("length") is not None:
        return float(d["length"])
    (x1, y1), (x2, y2) = node_xy(g, u), node_xy(g, v)
    return math.hypot(x2 - x1, y2 - y1)


def build_mock_graph(n: int = 22, spacing: float = 5.5, goal: int = 20) -> nx.Graph:
    g = nx.Graph()
    for i in range(n):
        g.add_node(i, xy=(i * spacing, 0.0), label="road")
    g.nodes[goal]["label"] = "road intersection"
    for i in range(n - 1):
        g.add_edge(i, i + 1, length=spacing, kind="seq")
    for i in range(n - 2):
        g.add_edge(i, i + 2, length=2 * spacing, kind="shortcut")
    g.graph["goal_node"] = goal
    return g
