import math

import numpy as np

from vlnav.config import LocCfg
from vlnav.features import l2_normalize
from vlnav.loc_eval import evaluate_replay
from vlnav.localization import Localizer, cumulative_odo_distance

SP = 5.0
XS = np.arange(0, 101, SP)            # 21 locations along a 100 m road


def _world(seed=0, alias=True):
    rng = np.random.default_rng(seed)
    base = {d: l2_normalize(rng.normal(size=(len(XS), 384)).astype(np.float32)) for d in (0, 1)}
    if alias:
        base[0][16] = base[0][4]        # x=80 looks exactly like x=20 (outbound view)
    return rng, base


def _nodes_and_emb(rng, base, roots=(0, 1)):
    nodes, E = [], []
    for r in roots:
        i = 0
        for d, order in ((0, range(len(XS))), (1, range(len(XS) - 1, -1, -1))):
            for k in order:
                x = float(XS[k])
                nodes.append({"id": len(nodes), "root": r, "idx": i, "odo": [x + r * 0.3, 0.0, 0.0],
                              "gt": [x, 0.0, 0.0 if d == 0 else math.pi], "k": k, "d": d})
                E.append(l2_normalize(base[d][k] + 0.12 * rng.normal(size=384).astype(np.float32)))
                i += 1
    return nodes, np.array(E, np.float32)


def _localizer(nodes, E, cfg=None):
    return Localizer(E, np.array([n["odo"][:2] for n in nodes]), np.array([n["root"] for n in nodes]),
                     np.array([n["idx"] for n in nodes]), [n["id"] for n in nodes], cfg or LocCfg(), SP)


def _query(rng, base, scale=1.03, bias=0.8, direction=0):
    order = range(len(XS)) if direction == 0 else range(len(XS) - 1, -1, -1)
    q = []
    for i, k in enumerate(order):
        x = float(XS[k])
        q.append({"id": 1000 + i, "root": 2, "idx": i, "odo": [scale * x + bias, 0.0, 0.0],
                  "gt": [x, 0.0, 0.0 if direction == 0 else math.pi],
                  "emb": l2_normalize(base[direction][k] + 0.12 * rng.normal(size=384).astype(np.float32))})
    return q


def test_cumulative_distance_per_root():
    xy = np.array([[0, 0], [3, 4], [3, 4], [0, 0], [10, 0]], float)
    d = cumulative_odo_distance(xy, np.array([0, 0, 0, 1, 1]), np.array([0, 1, 2, 0, 1]))
    assert np.allclose(d, [0, 5, 5, 0, 10])


def test_gating_blocks_aliased_far_node_and_ungated_does_not():
    rng, base = _world()
    nodes, E = _nodes_and_emb(rng, base)
    loc = _localizer(nodes, E)
    q = _query(rng, base)
    res = [loc.step(f["emb"], f["odo"][:2]) for f in q]
    assert res[0].fix is None and res[2].fix is None and res[3].fix is not None      # 4-frame vote
    fixes = [(f, r.fix) for f, r in zip(q, res) if r.fix]
    assert all(abs(nodes[fx.index]["gt"][0] - f["gt"][0]) <= 8.0 for f, fx in fixes)
    # ungated retrieval is fooled by the aliased pair at x=20 / x=80 for at least one of them
    k20, k80 = 4, 16
    wrong = [abs(nodes[res[k].raw_top1]["gt"][0] - q[k]["gt"][0]) > 8.0 for k in (k20, k80)]
    assert any(wrong)
    # gated top-1 at those frames is never the far twin
    assert all(abs(nodes[res[k].top1]["gt"][0] - q[k]["gt"][0]) <= 8.0 for k in (k20, k80))


def test_fix_reanchors_odometry_and_resets_uncertainty():
    rng, base = _world(alias=False)
    nodes, E = _nodes_and_emb(rng, base)
    loc = _localizer(nodes, E)
    q = _query(rng, base, scale=1.05, bias=1.0)
    s_before = None
    for f in q[:8]:
        r = loc.step(f["emb"], f["odo"][:2])
        if r.fix:
            assert loc.dist_since_fix == 0.0 and loc.fixed
            s_before = r.sigma_m if s_before is None else s_before
    # after fixes the corrected prediction tracks the truth far better than raw odometry
    pred_err = abs(q[7]["odo"][0] + loc.offset[0] - q[7]["gt"][0])
    raw_err = abs(q[7]["odo"][0] - q[7]["gt"][0])
    assert pred_err < raw_err and pred_err <= 5.0


def test_sequence_vote_rejects_inconsistent_frames():
    rng, base = _world(alias=False)
    nodes, E = _nodes_and_emb(rng, base)
    cfg = LocCfg(gate_sigma_k=50.0)        # effectively no gate so the vote alone must reject
    loc = _localizer(nodes, E, cfg)
    q = _query(rng, base)
    # corrupt frame 2 so it matches a node ~50 m away
    q[2]["emb"] = l2_normalize(base[0][14] + 0.01 * rng.normal(size=384).astype(np.float32))
    res = [loc.step(f["emb"], f["odo"][:2]) for f in q[:6]]
    assert all(r.fix is None for r in res[:5])            # window [0..3],[1..4],[2..5] all contain frame 2


def test_lost_flag_after_no_fix():
    rng, base = _world(alias=False)
    nodes, E = _nodes_and_emb(rng, base)
    loc = _localizer(nodes, E, LocCfg(lost_after_frames=3))
    junk = l2_normalize(rng.normal(size=384).astype(np.float32))
    flags = [loc.step(junk, [float(i * 40), 0.0]).lost for i in range(6)]
    assert flags[-1] is True and flags[0] is False


def test_replay_metrics_end_to_end_synthetic():
    rng, base = _world()
    nodes, E = _nodes_and_emb(rng, base)
    loc = _localizer(nodes, E)
    q = _query(rng, base)
    res = [loc.step(f["emb"], f["odo"][:2]) for f in q]
    rep = evaluate_replay(nodes, q, res, LocCfg())
    assert rep["n_fixes"] >= 15 and rep["first_fix_frame"] == 3
    assert rep["within_8m"] == 1.0 and rep["within_one_node"] >= 0.9
    assert rep["gated_top1_within_8m"] >= rep["ungated_top1_within_8m"]
    assert rep["n_failures"] == len(rep["failures"])


def test_eval_flags_wrong_leg_as_right_place_wrong_node():
    # fix lands on the opposite-leg node at the same place: within 8 m, but not within one node (heading differs)
    nodes = [{"id": 0, "root": 0, "idx": 0, "gt": [10.0, 0.0, 0.0]},
             {"id": 1, "root": 0, "idx": 5, "gt": [10.0, 0.0, math.pi]}]
    q = [{"id": 9, "root": 2, "idx": 0, "gt": [10.0, 0.0, 0.0]}]
    from vlnav.localization import Fix, LocResult
    r = LocResult(Fix(1, 1, (10.0, 0.0), 0.0, 0.9), 1, 1, 2, 10.0, 1.0, False)
    rep = evaluate_replay(nodes, q, [r], LocCfg())
    assert rep["within_8m"] == 1.0 and rep["within_one_node"] == 0.0 and rep["right_place_wrong_node"] == 1
    assert rep["n_failures"] == 1
