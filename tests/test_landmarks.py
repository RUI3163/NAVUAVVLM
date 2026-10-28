import math

from vlnav.landmarks import (build_landmarks, heading_deg, initial_road_heading, load_landmarks, polyline_length,
                             save_landmarks, turn_left_deg, wrap180)


def test_polyline_345():
    assert math.isclose(polyline_length([(0, 0), (3, 4), (3, 4), (6, 8)]), 10.0)


def test_headings_clockwise_from_north():
    assert heading_deg((0, 0), (1, 0)) == 0.0 and math.isclose(heading_deg((0, 0), (0, 1)), 90.0)
    assert math.isclose(heading_deg((0, 0), (0, -1)), 270.0)


def test_turn_left_sign_and_wrap():
    assert math.isclose(turn_left_deg(0.0, 270.0), 90.0)      # road is west of a north-facing drone -> turn left 90
    assert math.isclose(turn_left_deg(0.0, 90.0), -90.0)      # road to the east -> turn right
    assert math.isclose(turn_left_deg(350.0, 20.0), -30.0)
    assert wrap180(190.0) == -170.0


def test_initial_heading_skips_short_segments():
    assert math.isclose(initial_road_heading([(0, 0), (1, 5), (0, 20)], min_seg_m=10.0), 90.0)


def test_build_and_roundtrip(tmp_path):
    lm = build_landmarks((0.0, 0.0, -1.2), 0.0, [(0.0, -10.0), (0.0, -30.0)], (0.0, -60.0), max_alt_m=10.5)
    assert math.isclose(lm["road_length_p0_to_x1_m"], 60.0) and math.isclose(lm["straight_line_p0_to_x1_m"], 60.0)
    assert math.isclose(lm["road_heading_deg"], 270.0) and math.isclose(lm["initial_turn_left_deg"], 90.0)
    save_landmarks(str(tmp_path / "l.yaml"), lm)
    assert load_landmarks(str(tmp_path / "l.yaml")) == lm
