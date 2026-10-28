"""M1:  python m1_survey.py [--yaw-sign 1|-1] [--alt 10]   -> runs/m1_<ts>/landmarks.yaml (+ trail.jsonl, snapshots)"""
import argparse, asyncio, json, logging, os, sys
from datetime import datetime

from vlnav.config import Config
from vlnav.survey import run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaw-sign", type=float, default=1.0, help="-1 if 'j' turns the drone RIGHT")
    ap.add_argument("--alt", type=float, default=10.0)
    ap.add_argument("--sim-config-path", default=None)
    a = ap.parse_args()
    cfg = Config()
    if a.sim_config_path: cfg.sim.sim_config_path = a.sim_config_path
    out = os.path.join(cfg.runs_dir, f"m1_{datetime.now():%Y%m%d_%H%M%S}"); os.makedirs(out, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        handlers=[logging.FileHandler(os.path.join(out, "m1.log")), logging.StreamHandler(sys.stdout)])
    try:
        lm = asyncio.run(run(cfg, out, a.yaw_sign, a.alt))
    except Exception as exc:
        logging.getLogger("m1").error("M1 FAILED: %s", exc); return 1
    print(json.dumps(lm, indent=2)); return 0


if __name__ == "__main__":
    sys.exit(main())
