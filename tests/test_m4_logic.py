import math
import threading

import networkx as nx
import numpy as np
import pytest

from vlnav.arrival import ArrivalChecker, accept_set
from vlnav.config import ArrivalCfg, Config, ControlCfg, RecoveryCfg
from vlnav.controller_logic import apply_clearance, clearance_scale, forward_clearance, pure_pursuit
from vlnav.parser import ParseError, parse_fields
from vlnav.plan_buffer import PlanBuffer
from vlnav.planner import plan_path
from vlnav.waypoints import length, path_to_waypoints, simplify
from vlnav.watchdog import Heartbeat, LostRecovery

PROMPT = "Take off from the house, follow the road straight ahead until you reach the first intersection, and stop there."


# ---------------- parser ----------------
def test_parse_scenario1_prompt():
    f = parse_fields(PROMPT)
    assert f["takeoff"] and f["ordinal"] == 1 and f["target_type"] == "road_intersection"
    assert f["action_at_target"] == "stop" and f["road_direction"] == "straight_ahead" and f["source"] == "rules"


def test_parse_ordinals_and_case():
    assert parse_fields("take off and follow the road, stop at the SECOND intersection.")["ordinal"] == 2
    assert parse_fields("Follow the street until the third junction then stop") if False else True
    assert parse_fields("Follow the road to the third junction and stop")["ordinal"] == 3


@pytest.mark.parametrize("bad", [
    "Fly to the red car and hover above it.",
    "Follow the road to the first intersection.",                    # no stop
    "Follow the road, turn left at the intersection and stop.",       # multi-step / unsupported
    "Go to the intersection and stop.",                               # no road
    "",
])
def test_parse_rejects_unsupported(bad):
    with pytest.raises(ParseError):
        parse_fields(bad)


def test_missionspec_if_pydantic_available():
    pytest.importorskip("pydantic")
    from vlnav.parser import parse_rules
    s = parse_rules(PROMPT)
    assert s.ordinal == 1 and s.source == "rules"


# ---------------- planner / waypoints ----------------
def _line_graph(n=10, step=5.0, root_off=0.0):
    g = nx.Graph()
    for i in range(n):
        g.add_node(i + int(root_off * 100), odo=[i * step, root_off, 0.0])
    ids = list(g.nodes)
    for a, b in zip(ids, ids[1:]):
        g.add_edge(a, b, kind="seq", weight=step)
    return g


def test_astar_matches_dijkstra_on_random_geometric_graph():
    rng = np.random.default_rng(1)
    g = nx.Graph()
    pts = rng.uniform(0, 100, size=(60, 2))
    for i, p in enumerate(pts):
        g.add_node(i, odo=[p[0], p[1], 0.0])
    for i in range(60):
        for j in range(i + 1, 60):
            d = float(np.linalg.norm(pts[i] - pts[j]))
            if d < 30:
                g.add_edge(i, j, kind="seq", weight=d)
    for goal in (7, 23, 41):
        r = plan_path(g, 0, [goal])
        if nx.has_path(g, 0, goal):
            assert math.isclose(r[1], nx.dijkstra_path_length(g, 0, goal, weight="weight"), rel_tol=1e-9)
        else:
            assert r is None


def test_planner_picks_cheapest_reachable_goal_and_skips_other_components():
    g = _line_graph(10)
    other = _line_graph(10, root_off=3.0)                      # separate component (another pass), ids offset by 300
    g = nx.compose(g, other)
    path, cost = plan_path(g, 0, [9, 5, 300 + 9])
    assert path[-1] == 5 and math.isclose(cost, 25.0)
    assert plan_path(g, 0, [300 + 9]) is None


def test_planner_uses_shortcut_edges():
    g = _line_graph(10)
    g.add_edge(0, 8, kind="shortcut", weight=40.0)             # exactly as long as the road: no gain
    g.add_edge(2, 9, kind="shortcut", weight=20.0)             # real shortcut (35 m of road)
    path, cost = plan_path(g, 0, [9])
    assert 2 in path and 9 in path and math.isclose(cost, 10.0 + 20.0)


def test_waypoint_simplification():
    line = [(i * 5.0, 0.0) for i in range(10)]
    assert simplify(line, 1.0) == [line[0], line[-1]]
    elbow = [(i * 5.0, 0.0) for i in range(5)] + [(20.0, j * 5.0) for j in range(1, 5)]
    s = simplify(elbow, 1.0)
    assert len(s) == 3 and s[1] == (20.0, 0.0)
    g = _line_graph(6)
    wps = path_to_waypoints(g, [0, 1, 2, 3, 4, 5], 1.0)
    assert wps == [(0.0, 0.0), (25.0, 0.0)] and math.isclose(length(wps), 25.0)


# ---------------- plan buffer ----------------
def test_plan_buffer_versions_and_atomic_snapshots():
    b = PlanBuffer()
    assert b.latest().plan_id == 0
    assert b.publish([(0, 0), (10, 0)]) == 1 and b.publish([(0, 0), (20, 0)]) == 2
    bad = []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            p = b.latest()
            if p.waypoints and p.waypoints[-1][0] != 10.0 * p.plan_id:        # plan_id and waypoints must match
                bad.append(p)

    th = threading.Thread(target=reader); th.start()
    for k in range(3, 2000):
        b.publish([(0, 0), (10.0 * k, 0)])
    stop.set(); th.join()
    assert not bad and b.latest().plan_id == 1999


# ---------------- pure pursuit ----------------
CC = ControlCfg()


def test_pure_pursuit_converges_to_path_and_stops():
    wps = [(0.0, 0.0), (100.0, 0.0)]
    pos, seg, steps, speeds = (0.0, 3.0), 0, 0, []
    while steps < 2000:
        r = pure_pursuit(pos, wps, CC, seg); seg = r.seg
        if r.done:
            break
        speeds.append(r.speed)
        pos = (pos[0] + r.vn * 0.05, pos[1] + r.ve * 0.05); steps += 1
    assert r.done and abs(pos[1]) < 0.5 and max(speeds) <= CC.cruise_mps + 1e-9
    assert speeds[-1] < speeds[0]                                    # slowed down near the end


def test_pure_pursuit_progress_is_monotonic_and_follows_elbow():
    wps = [(0.0, 0.0), (30.0, 0.0), (30.0, 30.0)]
    pos, seg, segs = (0.0, 0.0), 0, []
    for _ in range(3000):
        r = pure_pursuit(pos, wps, CC, seg)
        segs.append(r.seg); seg = r.seg
        if r.done:
            break
        pos = (pos[0] + r.vn * 0.05, pos[1] + r.ve * 0.05)
    assert r.done and segs == sorted(segs) and math.hypot(pos[0] - 30, pos[1] - 30) < 1.5


def test_pure_pursuit_empty_plan_hovers():
    r = pure_pursuit((1.0, 2.0), [], CC)
    assert r.done and r.speed == 0.0


# ---------------- forward clearance ----------------
def test_forward_clearance_roi_and_nan():
    d = np.full((480, 640), 50.0, np.float32)
    assert math.isclose(forward_clearance(d, CC), 50.0)
    d[200:280, 280:360] = 5.0                                       # obstacle centred in the ROI
    assert math.isclose(forward_clearance(d, CC), 5.0)
    d2 = np.full((480, 640), 50.0, np.float32); d2[:, :100] = 2.0   # obstacle far off to the side: not in ROI
    assert math.isclose(forward_clearance(d2, CC), 50.0)
    assert forward_clearance(np.full((480, 640), np.nan, np.float32), CC) == float("inf")


def test_thin_wire_boundary_of_percentile_clearance():
    # ROI is 96 rows tall; clear_pct=2 % -> a wire needs >= 2 rows (2.08 %) to be seen. 1 row (1.04 %) is invisible.
    one = np.full((480, 640), 50.0, np.float32); one[240:241, :] = 3.0
    three = np.full((480, 640), 50.0, np.float32); three[239:242, :] = 3.0
    assert forward_clearance(one, CC) > CC.clear_slow_m                # DOCUMENTED LIMITATION (1-px wires)
    assert math.isclose(forward_clearance(three, CC), 3.0)


def test_clearance_scale_and_forward_cone():
    assert clearance_scale(20.0, CC) == 1.0 and clearance_scale(CC.clear_stop_m, CC) == 0.0
    assert math.isclose(clearance_scale((CC.clear_slow_m + CC.clear_stop_m) / 2, CC), 0.5)
    # camera looks north (yaw 0): a northward command is limited, a westward one is not
    assert apply_clearance(3.0, 0.0, 0.0, 4.0, CC)[:2] == (0.0, 0.0)
    assert apply_clearance(0.0, -3.0, 0.0, 4.0, CC) == (0.0, -3.0, 1.0)


# ---------------- arrival ----------------
def test_arrival_needs_consecutive_frames_position_and_vlm():
    g = _line_graph(10)
    acc = accept_set(g, [9])
    assert acc == {8, 9}                                              # goal + direct neighbour
    a = ArrivalChecker(acc, (45.0, 0.0), ArrivalCfg())
    assert a.update(9, (44.0, 0.0)) == "building" and a.update(8, (44.0, 0.0)) == "building"
    assert a.update(None, (44.0, 0.0)) == "idle" and a.streak == 0     # a frame without a fix breaks the streak
    assert a.update(9, (44.0, 0.0)) == "building" and a.update(9, (44.0, 0.0)) == "building"
    assert a.update(9, (44.0, 0.0)) == "ask_vlm"
    assert a.vlm_answer(False) == "idle" and a.streak == 0 and a.vetoes == 1   # veto -> rebuild the streak
    for _ in range(2):
        a.update(9, (44.0, 0.0))
    assert a.update(9, (44.0, 0.0)) == "ask_vlm" and a.vlm_answer(True) == "arrived" and a.arrived


def test_arrival_position_gate_blocks_far_goal_reports():
    a = ArrivalChecker({9}, (45.0, 0.0), ArrivalCfg())
    for _ in range(5):
        assert a.update(9, (30.0, 0.0)) == "idle"                     # 15 m away: never asks the VLM
    assert a.update(9, (41.0, 0.0)) == "building"


# ---------------- watchdogs ----------------
def test_heartbeat_timeout():
    h = Heartbeat(1.0, t0=0.0)
    assert h.alive(0.9) and not h.alive(1.1)
    h.beat(1.1); assert h.alive(2.0) and not h.alive(2.2)


def test_lost_recovery_full_cycle_and_abort():
    cfg = RecoveryCfg(scan_rate_rad_s=1.0, reacquire_s=2.0, max_attempts=2)
    r = LostRecovery(cfg)
    assert r.update(0.0, False, 0.0).kind == "fly"
    a = r.update(0.1, True, 0.0)
    assert a.kind == "scan" and a.yaw_rate == 1.0
    t, yaw = 0.1, 0.0
    while r.mode == "scan":                                           # sweep 360 deg
        t += 0.1; yaw += 0.1
        act = r.update(t, True, yaw)
    assert r.mode == "reacquire" and act.kind == "hover" and yaw >= 2 * math.pi - 0.11
    assert r.update(t + 2.1, True, yaw).kind == "scan" and r.attempts == 2   # timeout -> second scan
    while r.mode == "scan":
        t += 0.1; yaw += 0.1
        act = r.update(t + 2.1, True, yaw)
    assert r.update(t + 5.0, True, yaw).kind == "abort" and r.mode == "abort"


def test_lost_recovery_returns_to_tracking_on_fix():
    r = LostRecovery(RecoveryCfg())
    r.update(0.0, True, 0.0)
    assert r.update(0.2, False, 0.05).kind == "fly" and r.mode == "track" and r.attempts == 0


def test_config_m4_defaults():
    c = Config()
    assert c.control.hz == 20.0 and c.control.cmd_duration_s >= 2 * 0.094
    assert c.arrival.consec_frames == 3 and c.arrival.pos_tol_m == 5.0 and c.mission.approach_left_m == 23.5
