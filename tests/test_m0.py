import numpy as np
import pytest

from vlnav.config import CameraCfg, Config
from vlnav.frameio import depth_from_array, load_frame, rgb_from_msg, save_frame
from vlnav.jsonc import strip_comments
from vlnav.preflight import check_camera_config


def test_camera_defaults_match_spec():
    c = CameraCfg()
    assert (c.width, c.height, c.hfov_deg, c.roll_deg, c.pitch_deg, c.yaw_deg) == (640, 480, 90.0, 0.0, 0.0, 0.0)


def test_shipped_jsonc_matches_config():
    assert check_camera_config(Config()) == []


def test_preflight_catches_mismatch():
    from dataclasses import replace
    cfg = Config(); cfg.camera = replace(cfg.camera, width=1024, height=1024, pitch_deg=-10.0)
    p = check_camera_config(cfg)
    assert any("size" in s for s in p) and any("rpy" in s for s in p)


def test_strip_comments_keeps_strings():
    assert strip_comments('{"a": "x//y", // c\n "b": 1 /* d */}').replace(" ", "").replace("\n", "") == '{"a":"x//y","b":1}'


def test_depth_inf_becomes_nan_float32():
    d = depth_from_array(np.array([[1.0, np.inf], [np.nan, 5.0]], dtype=np.float16))
    assert d.dtype == np.float32 and np.isnan(d).sum() == 2 and d[0, 0] == 1.0


def test_rgb_decode_drops_alpha_and_swaps():
    raw = np.zeros((4, 4, 4), np.uint8); raw[..., 0] = 10; raw[..., 2] = 30; raw[..., 3] = 255
    a = rgb_from_msg(None, unpack_image=lambda _m: raw)
    b = rgb_from_msg(None, unpack_image=lambda _m: raw, swap_rgb=True)
    assert a.shape == (4, 4, 3) and a[0, 0, 0] == 10 and b[0, 0, 0] == 30


def test_save_load_roundtrip_identical(tmp_path):
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    dep = rng.uniform(1, 50, (480, 640)).astype(np.float32); dep[0, 0] = np.nan
    save_frame(str(tmp_path), "k0", rgb, dep)
    r2, d2 = load_frame(str(tmp_path), "k0")
    assert np.array_equal(rgb, r2) and np.array_equal(dep, d2, equal_nan=True)


def test_depth_sky_sentinel_finite_becomes_nan():
    d = depth_from_array(np.array([[6.3, 16416.0], [99.0, 101.0]], dtype=np.float16), max_m=100.0)
    assert d[0, 0] == np.float32(np.float16(6.3)) and np.isnan(d[0, 1]) and not np.isnan(d[1, 0]) and np.isnan(d[1, 1])
