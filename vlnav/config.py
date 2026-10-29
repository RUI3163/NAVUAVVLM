"""Single source of truth for every tunable. Frames: NED (x north, y east, z down); altitude = -z."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class CameraCfg:
    camera_id: str = "FrontCamera"      # sensor id in sim_config/robot_quadrotor_vlnav.jsonc
    width: int = 640
    height: int = 480
    hfov_deg: float = 90.0
    roll_deg: float = 0.0               # fixed
    pitch_deg: float = 0.0              # fixed (forward-facing)
    yaw_deg: float = 0.0                # fixed
    mount_xyz: Tuple[float, float, float] = (0.3, 0.0, 0.0)
    swap_rgb: bool = False              # set True if saved frames show swapped colours
    depth_max_m: float = 100.0          # depth above this (sky arrives as ~16416 m, finite) is treated as no-hit -> NaN


@dataclass
class SimCfg:
    scene_config: str = "scene_basic_drone.jsonc"
    sim_config_path: Optional[str] = None   # folder DIRECTLY containing the scene jsonc (default: ./sim_config)
    vehicle_name: str = "Drone1"
    delay_after_load_sec: int = 2
    imu_id: str = "IMU1"
    pose_wait_s: float = 5.0


@dataclass
class FlightCfg:
    takeoff_alt_m: float = 10.0
    alt_tol_m: float = 0.5
    climb_timeout_s: float = 60.0
    hover_s: float = 5.0
    ground_z_world: Optional[float] = None  # world NED z of the ground; None = first pose seen


@dataclass
class OdomCfg:
    """Dead reckoning from COMMANDED world velocity + gyro yaw, with injected seeded noise (per flight)."""
    scale_sigma: float = 0.015            # speed scale error ~ N(0, 1.5 %)
    heading_bias_sigma_deg: float = 1.0   # constant heading bias ~ N(0, 1 deg)
    gyro_yaw_sign: float = 1.0            # flip if the IMU z-rate integrates the wrong way


@dataclass
class MapCfg:
    keyframe_dist_m: float = 5.0
    keyframe_yaw_deg: float = 30.0
    shortcut_factor: float = 1.5          # tau = factor * median sequential node spacing (measured, not fixed)
    shortcut_scope: str = "same_root"     # "same_root": pairs within one flight | "any": across flights too
    n_passes: int = 3                     # each pass = out-and-back round trip (both directions)
    pass_lateral_offsets_m: Tuple[float, ...] = (0.0, 2.0, -2.0)
    cruise_mps: float = 3.0
    overshoot_m: float = 15.0             # fly this far past X1 before turning back
    turn_rate_dps: float = 30.0
    record_turn: bool = True
    held_out_pass: int = 2                # pass index reserved for the M3 replay test
    step_s: float = 0.2                   # command duration of the scripted flight loop


@dataclass
class FeatureCfg:
    model_id: str = "facebook/dinov2-small"   # DINOv2 ViT-S/14, 384-d
    input_w: int = 224                        # 640x480 -> 224x168, both multiples of 14, no crop
    input_h: int = 168
    pooling: str = "cls"                      # "cls" | "mean" (mean of patch tokens)
    device: str = "auto"                      # "auto" | "cuda" | "cpu"
    fp16: bool = True
    batch_size: int = 8
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class LocCfg:
    """Localisation. Lengths are multiples of the MEASURED median node spacing (meta.json), not fixed metres."""
    top_k: int = 5
    vote_frames: int = 4                 # sequence vote: this many consistent frames before a fix is accepted
    vote_spread_factor: float = 1.5      # max spread of odometry-transported votes, x spacing
    gate_floor_factor: float = 1.0       # gate radius = floor + k*sigma, floor in x spacing
    gate_sigma_k: float = 2.0
    drift_frac: float = 0.03             # odometry error per metre flown; M2 measured max 5.5 m over <=224 m (<=2.5 %)
    sigma_start_m: float = 1.0           # position sigma at P0 (known start)
    sigma_fix_factor: float = 0.5        # position sigma right after a fix, x spacing
    lost_after_frames: int = 8           # frames without an accepted fix before `lost` is reported
    match_radius_m: float = 8.0          # evaluation: "within 8 m"
    same_heading_deg: float = 60.0       # evaluation: heading window for the ground-truth nearest node
    colocate_m: float = 1.0              # evaluation: nodes this close (ground truth) are co-located
    colocate_heading_deg: float = 15.0   # ...and count as ONE place only if headings differ by less than this (half a keyframe yaw step)


@dataclass
class ControlCfg:
    """Control loop. cmd_duration_s is >2x the max command gap measured with image fetches running (94 ms)."""
    hz: float = 20.0
    cmd_duration_s: float = 0.2           # commands self-expire (measured: drone stops after expiry) -> fail-safe
    cruise_mps: float = 3.0
    lookahead_m: float = 6.0
    arrive_tol_m: float = 1.0
    min_speed_mps: float = 0.8
    end_slow_gain: float = 0.6            # speed = gain * distance to the end of the path, inside the slowdown zone
    roi_w_frac: float = 0.25              # forward-clearance ROI: central fraction of the depth image
    roi_h_frac: float = 0.20
    clear_pct: float = 2.0                # low percentile of ROI depth (thin wires may not show up in depth!)
    clear_slow_m: float = 15.0            # full speed above this clearance
    clear_stop_m: float = 6.0             # speed 0 at or below this clearance
    forward_cone_cos: float = 0.7         # clearance applies when the command is within ~45 deg of the camera axis
    hb_timeout_s: float = 1.0             # hover if the mission-manager heartbeat is older than this


@dataclass
class ArrivalCfg:
    consec_frames: int = 3                # localiser must report the goal (or neighbour) this many frames in a row
    pos_tol_m: float = 5.0                # corrected odometry within this of the goal node


@dataclass
class RecoveryCfg:
    scan_rate_rad_s: float = 0.5          # yaw rate while re-localising (360 deg takes ~12.6 s)
    reacquire_s: float = 5.0
    max_attempts: int = 3


@dataclass
class MissionCfg:
    prompt: str = "Take off from the house, follow the road straight ahead until you reach the first intersection, and stop there."
    # SCENARIO PRIOR (hand-fed, not sensed): the road lies this far to the LEFT (west) of the take-off point.
    # Executed open-loop from commanded velocity (never ground truth). Value from the M1 survey (23.5 m).
    approach_left_m: float = 23.5
    approach_speed_mps: float = 3.0
    waypoint_tol_m: float = 1.0           # Douglas-Peucker tolerance when simplifying the planned path


@dataclass
class Config:
    camera: CameraCfg = field(default_factory=CameraCfg)
    sim: SimCfg = field(default_factory=SimCfg)
    flight: FlightCfg = field(default_factory=FlightCfg)
    odom: OdomCfg = field(default_factory=OdomCfg)
    map: MapCfg = field(default_factory=MapCfg)
    features: FeatureCfg = field(default_factory=FeatureCfg)
    loc: LocCfg = field(default_factory=LocCfg)
    control: ControlCfg = field(default_factory=ControlCfg)
    arrival: ArrivalCfg = field(default_factory=ArrivalCfg)
    recovery: RecoveryCfg = field(default_factory=RecoveryCfg)
    mission: MissionCfg = field(default_factory=MissionCfg)
    logs_dir: str = "logs"
    runs_dir: str = "runs"
    seed: int = 0
