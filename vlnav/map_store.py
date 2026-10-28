"""Save / reload the map: nodes.json, edges.json, embeddings.npy, meta.json (images stay where frameio saved them)."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np


def save_map(d: str, nodes: List[dict], emb: np.ndarray, g: nx.Graph, meta: Dict) -> None:
    os.makedirs(d, exist_ok=True)
    assert emb.shape[0] == len(nodes), "one embedding per node"
    json.dump(nodes, open(os.path.join(d, "nodes.json"), "w"))
    json.dump([[u, v, dict(a)] for u, v, a in g.edges(data=True)], open(os.path.join(d, "edges.json"), "w"))
    np.save(os.path.join(d, "embeddings.npy"), emb.astype(np.float32))
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=2)


def load_map(d: str) -> Tuple[List[dict], np.ndarray, nx.Graph, Dict]:
    nodes = json.load(open(os.path.join(d, "nodes.json")))
    g = nx.Graph()
    for n in nodes:
        g.add_node(n["id"], **{k: v for k, v in n.items() if k != "id"})
    for u, v, a in json.load(open(os.path.join(d, "edges.json"))):
        g.add_edge(u, v, **a)
    return nodes, np.load(os.path.join(d, "embeddings.npy")), g, json.load(open(os.path.join(d, "meta.json")))
