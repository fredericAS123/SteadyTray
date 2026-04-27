import argparse
import os
import sys
import time
from collections import deque

import mujoco
import mujoco.viewer
import numpy as np
import torch


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.config import Config
from scripts.mujoco_sim import (
    build_stage3_teacher_encoder_obs,
    configure_viewer_camera,
    initialize_model_state,
    load_mujoco_model,
    make_policy_obs,
    parse_cmd,
    pd_control,
    resolve_repo_path,
)


def detect_stage3_encoder_obs_size(policy, default_size):
    try:
        state_dict = policy.state_dict()
        for key, value in state_dict.items():
            if key.endswith("history_encoder.embed.weight") or key.endswith("encoder.embed.weight"):
                return int(value.shape[1])
    except Exception:
        pass
    return default_size


def run_stage3(policy_path, config, cmd, encoder_seq_len, duration=None, dry_run=False):
    policy_path = resolve_repo_path(policy_path)
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy file not found: {policy_path}")

    model = load_mujoco_model(config.xml_path, config.simulation_dt, config.remove_bodies)
    data = mujoco.MjData(model)
    target_dof_pos = initialize_model_state(model, data, config)
    action = np.zeros(config.num_actions, dtype=np.float32)

    policy = torch.jit.load(policy_path).eval()
    expected_encoder_dim = getattr(config, "stage3_teacher_encoder_obs_dim", 117)
    encoder_obs_dim = detect_stage3_encoder_obs_size(policy, expected_encoder_dim)
    if encoder_obs_dim != expected_encoder_dim:
        print(
            f"Warning: policy encoder dim is {encoder_obs_dim}, "
            f"but config builds {expected_encoder_dim}. The script will validate the first frame."
        )

    frame_stack = deque(maxlen=5)
    for _ in range(5):
        frame_stack.append(np.zeros(config.num_obs, dtype=np.float32))

    encoder_frame_stack = deque(maxlen=encoder_seq_len)
    for _ in range(encoder_seq_len):
        encoder_frame_stack.append(np.zeros(expected_encoder_dim, dtype=np.float32))

    def compute_once():
        nonlocal action, target_dof_pos
        qj = data.qpos[7 : 7 + config.num_actions].astype(np.float32)
        dqj = data.qvel[6 : 6 + config.num_actions].astype(np.float32)
        quat = data.qpos[3:7].astype(np.float32)
        omega = data.qvel[3:6].astype(np.float32)

        policy_obs = make_policy_obs(frame_stack, qj, dqj, quat, omega, cmd, action, config)
        encoder_obs = build_stage3_teacher_encoder_obs(data, model, qj, dqj, quat, omega, cmd, action, config)
        if encoder_obs.shape[0] != encoder_obs_dim:
            raise ValueError(f"Stage 3 encoder obs has {encoder_obs.shape[0]} dims, policy expects {encoder_obs_dim}")

        encoder_frame_stack.append(encoder_obs)
        encoder_array = np.array(encoder_frame_stack, dtype=np.float32)
        encoder_tensor = torch.from_numpy(encoder_array).unsqueeze(1)
        policy_tensor = torch.from_numpy(policy_obs).unsqueeze(0)

        with torch.no_grad():
            action_policy = policy(encoder_tensor, policy_tensor).squeeze(0).numpy()
        if action_policy.shape[0] != config.num_actions:
            raise ValueError(f"Policy returned {action_policy.shape[0]} actions, expected {config.num_actions}")

        action = action_policy[config.policy_to_robot]
        target_dof_pos = action * config.action_scale[config.policy_to_robot] + config.default_angles[config.policy_to_robot]

    if dry_run:
        compute_once()
        print(f"Dry run OK: policy={policy_path}, xml={config.xml_path}, encoder_dim={encoder_obs_dim}")
        return

    sim_duration = config.simulation_duration if duration is None else duration
    counter = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        configure_viewer_camera(viewer)
        start = time.time()
        while viewer.is_running() and time.time() - start < sim_duration:
            step_start = time.time()
            tau = pd_control(
                target_dof_pos,
                data.qpos[7 : 7 + config.num_actions],
                config.kps,
                np.zeros_like(config.kds),
                data.qvel[6 : 6 + config.num_actions],
                config.kds,
            )
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)

            counter += 1
            if counter % config.control_decimation == 0:
                compute_once()

            viewer.sync()
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


def main():
    parser = argparse.ArgumentParser(description="Stage 3 object teacher/adapter sim2sim in MuJoCo")
    parser.add_argument("--policy", type=str, required=True, help="Path to exported Stage 3 adapter JIT policy")
    parser.add_argument(
        "--config",
        type=str,
        default="deploy/configs/g1_stage3_object_teacher.yaml",
        help="Path to Stage 3 MuJoCo config",
    )
    parser.add_argument("--duration", type=float, default=None, help="Override simulation duration in seconds")
    parser.add_argument("--cmd", type=float, nargs=3, default=None, metavar=("VX", "VY", "WZ"))
    parser.add_argument("--encoder_seq_len", type=int, default=32, help="Stage 3 encoder history length")
    parser.add_argument("--dry-run", action="store_true", help="Load model/policy and run one inference without viewer")
    args = parser.parse_args()

    config = Config(args.config)
    cmd = parse_cmd(args.cmd, config.cmd_init)
    run_stage3(args.policy, config, cmd, args.encoder_seq_len, duration=args.duration, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
