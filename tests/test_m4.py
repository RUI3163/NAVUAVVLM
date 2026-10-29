import asyncio
import threading
import time

import networkx as nx
import numpy as np
import pytest

from vlnav.controller import ControllerConfig, ControlLoop, PurePursuit, forward_clearance
from vlnav.graph_utils import build_mock_graph, node_xy
from vlnav.mission import MissionConfig, MissionManager, State, plan_route
from vlnav.mock_sim import MockLocalizer, MockSim
from vlnav.odometry import DeadReckoning, OdomNoise
from vlnav.parser import GroundingError, ground_to_node, parse_instruction, verify_intersection_rule
from vlnav.perception import Frame, LocFix
from vlnav.plan_buffer import Heartbeat, LatestSlot, Mode, PlanBuffer
from vlnav.runtime import M4Config, run_mission
from vlnav.vlm_worker import VLMClient, VLMConfig, VLMKind

PROMPT = "Take off from the house, follow the road straight ahead until you reach the first intersection, and stop there."


# ---------------- plan buffer
def test_plan_ids_monotonic_and_immutable():
    pb = PlanBuffer()
    p1 = pb.publish([(0, 0), (1, 0)])
    p2 = pb.publish([(0, 0)], Mode.HOVER)
    assert p2.plan_id == p1.plan_id + 1 and pb.snapshot() is p2
    with pytest.raises(Exception):
        p1.plan_id = 5  # frozen


def test_plan_buffer_no_torn_reads_under_contention():
    pb, bad, stop = PlanBuffer(), [], threading.Event()

    def writer():
        for i in range(3000):
            pb.publish([(i, i)] * (i % 5 + 1), reason=str(i))
        stop.set()

    def reader():
        last = 0
        while not stop.is_set():
            p = pb.snapshot()
            if p.plan_id < last or (p.waypoints and any(w != p.waypoints[0] for w in p.waypoints)):
                bad.append(p)
            last = p.plan_id

    ts = [threading.Thread(target=reader) for _ in range(3)] + [threading.Thread(target=writer)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert not bad


def test_latest_slot_drops_stale():
    s = LatestSlot()
    for i in range(5):
        s.put(i)
    item, v = s.take_if_newer(0)
    assert item == 4 and s.take_if_newer(v)[0] is None


# ---------------- A*
def test_astar_prefers_shortcuts_and_respects_weights():
    g = build_mock_graph(n=11, goal=10)
    r = plan_route(g, 0, 10)
    assert r[0] == 0 and r[-1] == 10 and len(r) <= 6  # shortcuts exploited (cost equal => any min path)
    g.remove_edge(2, 4); g.remove_edge(3, 4)  # force detour around node 4
    g.edges[0, 2]["length"] = 1000
    r2 = plan_route(g, 0, 10)
    assert 4 not in r2 or r2[r2.index(4) - 1] in g[4]
    assert nx.path_weight(g, r2, "length") <= nx.path_weight(g, [0, 1, 2, 3, 5, 6, 7, 8, 9, 10], "length") + 1e-6


def test_astar_no_route_raises():
    g = build_mock_graph(n=6, goal=5); g.remove_nodes_from([2, 3])
    with pytest.raises(nx.NetworkXNoPath):
        plan_route(g, 0, 5)


# ---------------- controller + watchdog
def _rig(wps=((0, 0), (50, 0)), hb_age=0.0, depth=200.0, yaw=0.0):
    cfg = ControllerConfig()
    odom = DeadReckoning(OdomNoise(0, 0, 0), 0.0, 0.0, yaw)
    plans, hb, frames, cmds = PlanBuffer(), Heartbeat(clock=lambda: 100.0 - hb_age), LatestSlot(), LatestSlot()
    plans.publish(wps)
    frames.put(Frame(1, 100.0, np.zeros((4, 4, 3)), np.full((48, 64), depth, np.float32)))
    return ControlLoop(cfg, plans, odom, frames, cmds, hb), hb


def test_pure_pursuit_forward_and_turn():
    c, _ = _rig()
    cmd, why = c.step(100.0)
    assert cmd.vx > 0 and abs(cmd.yaw_rate) < 1e-6 and why == "follow"
    c, _ = _rig(yaw=np.pi / 2)  # facing 90 deg off path -> turn in place
    cmd, _ = c.step(100.0)
    assert cmd.vx == 0 and cmd.yaw_rate < 0


def test_watchdog_heartbeat_forces_hover():
    c, hb = _rig(hb_age=0.5)
    assert c.step(100.0)[0].vx > 0
    c, hb = _rig(hb_age=1.5)
    cmd, why = c.step(100.0)
    assert cmd.vx == 0 and cmd.yaw_rate == 0 and why == "watchdog:mission_heartbeat"


def test_clearance_stops_and_slows():
    assert _rig(depth=3.0)[0].step(100.0)[1] == "blocked:clearance"
    cmd, why = _rig(depth=10.0)[0].step(100.0)
    assert why == "follow:slow" and 0 < cmd.vx < ControllerConfig().v_max
    assert forward_clearance(np.full((8, 8), np.nan, np.float32), ControllerConfig()) == 100.0


def test_pure_pursuit_progress_is_monotonic():
    pp = PurePursuit(5.0)
    wps = [(0, 0), (10, 0), (10, 10)]
    t0, _ = pp.target(0, 0, wps, 1)
    t1, rem = pp.target(10, 5, wps, 1)
    assert t0 == (5.0, 0.0) and t1[1] > 5 and abs(rem - 5) < 1e-6


def test_control_loop_never_blocks_on_slow_vlm():
    """A VLM stuck in a 1.5 s call must not reduce the control rate."""
    async def go():
        c, hb = _rig()
        c.clock = time.monotonic; hb._clock = time.monotonic; hb.beat()
        c.frames.put(Frame(1, time.monotonic(), np.zeros((4, 4, 3)), np.full((48, 64), 200.0, np.float32)))
        vlm = VLMClient(VLMConfig(mode="rule"))
        slow = lambda req: (time.sleep(1.5), "YES")[1]
        vlm._rule.run = slow
        vlm.start(); vlm.submit(VLMKind.CAPTION, {"rgb": None})
        stop = asyncio.Event()
        t = asyncio.create_task(c.run(stop))
        await asyncio.sleep(1.0); stop.set(); await t; vlm.stop()
        return c.ticks
    assert asyncio.run(go()) >= 17  # ~20 Hz


# ---------------- parser fallback
def test_parse_scenario_1():
    p = parse_instruction(PROMPT)
    assert (p.start_anchor, p.behavior, p.direction, p.target_kind, p.ordinal, p.stop) == \
           ("house", "follow_road", "straight", "intersection", 1, True)


def test_grounding_label_ordinal_and_failure():
    g = build_mock_graph(goal=20); del g.graph["goal_node"]
    for n in (8, 9):
        g.nodes[n]["label"] = "road intersection"
    assert ground_to_node(parse_instruction(PROMPT), g, 0).node in (8, 9)          # 1st cluster
    p2 = parse_instruction(PROMPT.replace("first", "second"))
    assert ground_to_node(p2, g, 0).node == 20                                     # 2nd cluster
    with pytest.raises(GroundingError):
        ground_to_node(parse_instruction(PROMPT.replace("first", "third")), g, 0)


def test_rule_verifier_on_synthetic_depth():
    open_d = np.full((48, 64), 200.0, np.float32)
    closed = open_d.copy(); closed[:, :16] = closed[:, -16:] = 12.0
    assert verify_intersection_rule(open_d) is True
    assert verify_intersection_rule(closed) is False
    one_side = open_d.copy(); one_side[:, :16] = 12.0
    assert verify_intersection_rule(one_side) is False


# ---------------- mission logic (synchronous, deterministic clock)
class FakeVLM:
    def __init__(self, ans="YES"): self.ans, self.sub, self._r = ans, [], []
    def submit(self, kind, payload, key=None, ttl_s=None):
        self.sub.append(kind); rid = len(self.sub)
        if kind is VLMKind.ARRIVAL:
            from vlnav.vlm_worker import VLMResult
            self._r.append(VLMResult(rid, kind, self.ans, 0.1, "fake"))
        return rid
    def poll(self): r, self._r = self._r, []; return r
    def fallback_verify(self, d): return "YES"


def _mm(ans="YES", cfg=None):
    g = build_mock_graph()
    odom = DeadReckoning(OdomNoise(0, 0, 0), 0.0, 0.0, 0.0)
    fixes, frames, plans, hb = LatestSlot(), LatestSlot(), PlanBuffer(), Heartbeat()
    frames.put(Frame(1, 0.0, np.zeros((4, 4, 3)), np.full((48, 64), 200.0, np.float32)))
    vlm = FakeVLM(ans)
    mm = MissionManager(g, 0, 20, plans, hb, fixes, frames, odom, vlm, cfg or MissionConfig(caption_nodes=False))
    mm.start(0.0)
    return mm, g, odom, fixes, vlm, plans


def _fix(fixes, fid, node, status="fix"):
    fixes.put(LocFix(fid, node, status, 3.0, 0.0))


def test_arrival_requires_hits_distance_and_vlm_yes():
    mm, g, odom, fixes, vlm, plans = _mm()
    odom.reanchor(*node_xy(g, 19), 1.0)                      # within 5 m of goal(110,0)
    for i, n in enumerate((19, 20, 21), 1):                  # goal + 1-hop neighbours
        _fix(fixes, i, n); mm.tick(i * 0.1)
        assert mm.state is (State.NAVIGATE if i < 3 else State.VERIFY)
    mm.tick(0.4)                                             # VLM answer is polled on the next tick
    assert mm.success and plans.snapshot().mode is Mode.HOVER and VLMKind.ARRIVAL in vlm.sub


def test_no_arrival_if_far_or_two_hops():
    mm, g, odom, fixes, vlm, _ = _mm()
    for i in range(1, 6):
        _fix(fixes, i, 20); mm.tick(i * 0.1)                  # odom still ~0 m => 110 m away
    assert mm.state is State.NAVIGATE and not vlm.sub
    odom.reanchor(*node_xy(g, 20), 1.0)
    mm.hits = 0
    for i in range(10, 15):
        _fix(fixes, i, 18); mm.tick(i * 0.1)                  # 2 hops away: never counts
    assert mm.hits == 0 and mm.state is State.NAVIGATE


def test_verify_no_resumes_and_cooldown():
    mm, g, odom, fixes, vlm, plans = _mm(ans="NO")
    odom.reanchor(*node_xy(g, 20), 1.0)
    for i in (1, 2, 3, 4):
        _fix(fixes, i, 20); mm.tick(i * 0.1)
    assert mm.state is State.NAVIGATE and plans.snapshot().mode is Mode.FOLLOW
    for i in (4, 5, 6):
        _fix(fixes, i, 20); mm.tick(0.5 + i * 0.01)
    assert vlm.sub.count(VLMKind.ARRIVAL) == 1               # cooldown blocks re-verify


def test_verify_timeout_uses_rule_fallback():
    mm, g, odom, fixes, vlm, _ = _mm()
    vlm.poll = lambda: []                                    # VLM never answers
    odom.reanchor(*node_xy(g, 20), 1.0)
    for i in (1, 2, 3):
        _fix(fixes, i, 20); mm.tick(i * 0.1)
    assert mm.state is State.VERIFY
    mm.tick(10.0)
    assert mm.success and "timeout" in mm.verify_log[-1]["how"]


def test_lost_after_8_frames_sweeps_then_reanchors():
    mm, g, odom, fixes, vlm, plans = _mm()
    for i in range(1, 8):
        _fix(fixes, i, None, "searching"); mm.tick(i * 0.1)
    assert mm.state is State.NAVIGATE                        # 7 misses: not lost yet
    _fix(fixes, 8, None, "searching"); mm.tick(1.0)
    assert mm.state is State.LOST_SWEEP and plans.snapshot().mode is Mode.SWEEP
    _fix(fixes, 9, 5); mm.tick(2.0)
    assert mm.state is State.NAVIGATE and plans.snapshot().mode is Mode.FOLLOW


def test_lost_status_immediate_and_sweep_exhaustion_fails():
    mm, g, odom, fixes, vlm, plans = _mm()
    _fix(fixes, 1, None, "lost"); mm.tick(0.1)
    assert mm.state is State.LOST_SWEEP
    t = 0.1
    for _ in range(10):
        t += 13.0; mm.tick(t)
    assert mm.state is State.FAILED and plans.snapshot().mode is Mode.HOVER


def test_mission_manager_heartbeat_beats_each_tick():
    mm, *_ = _mm()
    mm.hb._clock = lambda: 5.0
    mm.tick(5.0)
    assert mm.hb.age(5.0) == 0.0


def test_replan_when_fix_off_route():
    mm, g, odom, fixes, vlm, plans = _mm()
    mm.route = [0, 2, 4]
    pid = plans.snapshot().plan_id
    _fix(fixes, 1, 7); mm.tick(0.1)
    assert plans.snapshot().plan_id == pid + 1 and mm.route[0] == 7


# ---------------- end-to-end offline
def test_e2e_offline_mock_success():
    g = build_mock_graph()
    sim = MockSim(0.0, 0.0, 0.0)
    cfg = M4Config(); cfg.controller.v_max = 20.0; cfg.controller.lookahead_m = 8.0
    cfg.odom_noise = OdomNoise(seed=1)
    cfg.perception.rate_hz = 10.0
    res = asyncio.run(run_mission(sim, g, 0, 20, MockLocalizer(sim, g), cfg, 0.0, 40.0))
    assert res.success, res.events
    assert abs(sim.true_xy[0] - 110.0) < 6.0
    assert res.control_hz > 15
