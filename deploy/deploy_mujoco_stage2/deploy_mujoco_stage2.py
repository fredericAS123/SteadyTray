import argparse
import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.config import Config
from scripts.mujoco_sim import parse_cmd, run_standard_stage


def main():
    parser = argparse.ArgumentParser(description="Stage 2 tray-holding sim2sim in MuJoCo")
    parser.add_argument("--policy", type=str, required=True, help="Path to Stage 2 JIT policy")
    parser.add_argument(
        "--config",
        type=str,
        default="deploy/configs/g1_stage2_tray.yaml",
        help="Path to Stage 2 MuJoCo config",
    )
    parser.add_argument("--duration", type=float, default=None, help="Override simulation duration in seconds")
    parser.add_argument("--cmd", type=float, nargs=3, default=None, metavar=("VX", "VY", "WZ"))
    parser.add_argument("--dry-run", action="store_true", help="Load model/policy and run one inference without viewer")
    args = parser.parse_args()

    config = Config(args.config)
    cmd = parse_cmd(args.cmd, config.cmd_init)
    run_standard_stage(args.policy, config, cmd=cmd, duration=args.duration, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
