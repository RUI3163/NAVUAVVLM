import math

import networkx as nx
import numpy as np
import pytest

from vlnav.config import Config, FeatureCfg, OdomCfg
from vlnav.features import l2_normalize, pool_tokens, preprocess
from vlnav.graph import build_graph, derive_tau, edge_counts, spacing_stats
from vlnav.map_eval import odometry_drift, shortcut_stats
from vlnav.map_store import load_map, save_map
from vlnav.mapping import KeyframePolicy
from vlnav.odometry import DeadReckoner


# ---------- odometry ----------
def test_odometry_zero_noise_exact():
    o = DeadReckoner(OdomCfg(scale_sigma=0.0, heading_bias_sigma_deg=0.0), seed=1)
    for _ in range(100):
        o.update(3.0, 0.0, 0.1)
    assert math.isclose(o.pose()[0], 30.0) and abs(o.pose()[1]) < 1e-9 and math.isclose(o.dist, 30.0)


def test_odometry_seeded_and_noisy():
    a, b, c = (DeadReckoner(OdomCfg(), seed=s) for s in (5, 5, 6))
    for o in (a, b, c):
        o.update(3.0, 0.0, 10.0)
    assert a.pose() == b.pose() and a.pose() != c.pose()
    assert abs(a.scale - 1) < 0.075 and abs(a.bias) < math.radians(5)


def test_odometry_heading_bias_rotates_velocity():
    o = DeadReckoner(OdomCfg(scale_sigma=0.0, heading_bias_sigma_deg=0.0), seed=0)
    o.bias = math.radians(10.0)
    o.update(10.0, 0.0, 1.0)
    assert math.isclose(math.atan2(o.e, o.n), math.radians(10.0))


def test_gyro_yaw_integration():
    o = DeadReckoner(OdomCfg(heading_bias_sigma_deg=0.0), seed=0, yaw0_rad=0.0)
    o.on_gyro(0.5, 100.0)
    for i in range(1, 11):
        o.on_gyro(0.5, 100.0 + 0.1 * i)
    assert math.isclose(o.pose()[2], 0.5, rel_tol=1e-6)


# ---------- keyframes ----------
def test_keyframe_distance_and_yaw_triggers():
    p = KeyframePolicy(5.0, 30.0)
    assert p.update(0, 0, 0.0) is True            # first frame
    assert p.update(4.9, 0, 0.0) is False
    assert p.update(5.0, 0, 0.0) is True
    assert p.update(5.0, 0, math.radians(29)) is False
    assert p.update(5.0, 0, math.radians(30)) is True


# ---------- graph ----------
def _out_and_back(drift=1.0, step=5.0, length=100.0, root=0, start_id=0):
    nodes, i = [], 0
    outb = np.arange(0, length + 1e-6, step)
    for x in list(outb) + list(outb[::-1][1:]):
        leg_back = len(nodes) >= len(outb)
        nodes.append({"id": start_id + i, "root": root, "idx": i, "odo": [x + (drift if leg_back else 0.0), 0.0, 0.0],
                      "gt": [x, 0.0, 0.0]})
        i += 1
    return nodes


def test_graph_sequential_and_shortcuts():
    nodes = _out_and_back()
    st = spacing_stats(nodes)
    assert math.isclose(st["median"], 5.0)
    tau = derive_tau(nodes, 1.5)
    assert math.isclose(tau, 7.5)
    g = build_graph(nodes, tau)
    c = edge_counts(g)
    assert c["seq"] == len(nodes) - 1
    # no shortcut between i and i+2 on one leg (10 m > tau); out/back pairs at the same place are linked
    assert not g.has_edge(0, 2)
    n_out = 21                                  # turn-around node is idx 20; return leg starts at idx 21
    assert g.has_edge(20 - 1, 20 + 2)           # outbound node 19 (x=95) <-> return node 22 (x=90+1 drift): 4-5 m apart
    assert any(d["kind"] == "shortcut" for _, _, d in g.edges(data=True))
    for u, v, d in g.edges(data=True):
        if d["kind"] == "shortcut":
            assert abs(nodes[u]["idx"] - nodes[v]["idx"]) > 1 and d["weight"] <= tau


def test_scope_same_root_vs_any():
    a = _out_and_back(root=0)
    b = _out_and_back(root=1, start_id=len(a))
    for n in b:
        n["idx"] = n["idx"]
    nodes = a + b
    tau = derive_tau(nodes, 1.5)
    g_same, g_any = build_graph(nodes, tau, "same_root"), build_graph(nodes, tau, "any")
    cross = lambda g: sum(1 for u, v, d in g.edges(data=True) if d["kind"] == "shortcut" and nodes[u]["root"] != nodes[v]["root"])
    assert cross(g_same) == 0 and cross(g_any) > 0
    with pytest.raises(ValueError):
        build_graph(nodes, tau, "nope")


def test_spacing_ignores_in_place_turn_nodes():
    nodes = [{"id": i, "root": 0, "idx": i, "odo": [5.0 * i, 0.0, 0.0], "gt": [5.0 * i, 0, 0]} for i in range(5)]
    nodes += [{"id": 5 + j, "root": 0, "idx": 5 + j, "odo": [20.0, 0.0, 0.0], "gt": [20.0, 0, 0]} for j in range(6)]
    assert math.isclose(spacing_stats(nodes)["median"], 5.0)


def test_shortcut_precision_against_gt():
    nodes = _out_and_back(drift=1.0)
    tau = derive_tau(nodes, 1.5)
    g = build_graph(nodes, tau)
    s = shortcut_stats(g, nodes, tau)
    assert s["n_shortcuts"] > 0 and s["precision_le_tau"] == 1.0
    # make GT disagree for one node -> precision drops
    nodes[-2]["gt"][0] += 100.0
    s2 = shortcut_stats(g, nodes, tau)
    assert s2["precision_le_tau"] < 1.0


def test_odometry_drift_report():
    nodes = _out_and_back(drift=2.0)
    d = odometry_drift(nodes)[0]
    assert math.isclose(d["max_err_m"], 2.0) and d["n"] == len(nodes)


def test_map_save_reload_identical(tmp_path):
    nodes = _out_and_back()
    g = build_graph(nodes, derive_tau(nodes, 1.5))
    emb = l2_normalize(np.random.default_rng(0).normal(size=(len(nodes), 384)).astype(np.float32))
    save_map(str(tmp_path), nodes, emb, g, {"tau_m": 7.5})
    n2, e2, g2, m2 = load_map(str(tmp_path))
    assert n2 == nodes and np.array_equal(e2, emb) and m2 == {"tau_m": 7.5}
    assert sorted(map(sorted, g.edges())) == sorted(map(sorted, g2.edges()))
    kinds = lambda gg: sorted((tuple(sorted(k)), v) for k, v in nx.get_edge_attributes(gg, "kind").items())
    assert kinds(g) == kinds(g2)


# ---------- features (torch-free parts) ----------
def test_preprocess_shape_dtype_and_no_crop():
    cfg = FeatureCfg()
    assert cfg.input_w % 14 == 0 and cfg.input_h % 14 == 0 and cfg.input_w / cfg.input_h == 640 / 480
    img = np.zeros((480, 640, 3), np.uint8); img[:, :, 2] = 255        # pure red in BGR
    x = preprocess(img, cfg)
    assert x.shape == (3, 168, 224) and x.dtype == np.float32
    assert np.allclose(x[0], (1.0 - 0.485) / 0.229, atol=1e-4) and np.allclose(x[1], (0 - 0.456) / 0.224, atol=1e-4)


def test_preprocess_keeps_full_field_of_view():
    img = np.zeros((480, 640, 3), np.uint8); img[:, :8] = 255          # white strip on the far-left edge
    assert preprocess(img, FeatureCfg())[0, :, 0].mean() > 0            # edge still present: no centre crop


def test_pool_tokens_and_l2():
    t = np.zeros((2, 1 + 4, 3), np.float32); t[:, 0] = [1, 2, 2]; t[:, 1:] = 3.0
    assert np.array_equal(pool_tokens(t, "cls"), t[:, 0]) and np.allclose(pool_tokens(t, "mean"), 3.0)
    with pytest.raises(ValueError):
        pool_tokens(t, "max")
    assert np.allclose(np.linalg.norm(l2_normalize(t[:, 0]), axis=1), 1.0)


def test_config_m2_defaults():
    c = Config()
    assert (c.map.keyframe_dist_m, c.map.keyframe_yaw_deg, c.map.shortcut_scope) == (5.0, 30.0, "same_root")
    assert (c.features.pooling, c.odom.scale_sigma) == ("cls", 0.015)
