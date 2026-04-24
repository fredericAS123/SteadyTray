"""Shared MuJoCo helpers for stage-by-stage sim2sim deployment."""

from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET
from collections import deque
from typing import Iterable

import mujoco
import mujoco.viewer
import numpy as np
import torch

from scripts.config import Config
from scripts.policy_runner import get_gravity_orientation, process_observation_with_frame_stack


torch.set_num_threads(1)
torch.set_num_interop_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculate PD torques from position commands."""
    return (target_q - q) * kp + (target_dq - dq) * kd


def parse_cmd(cmd_values, default_cmd):
    if cmd_values is None:
        return default_cmd
    return np.array(cmd_values, dtype=np.float32)


def resolve_repo_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(path)


def _remove_named_bodies(root: ET.Element, body_names: set[str]) -> None:
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "body" and child.attrib.get("name") in body_names:
                parent.remove(child)


def _make_meshdir_absolute(root: ET.Element, xml_path: str) -> None:
    compiler = root.find("compiler")
    if compiler is None:
        return
    meshdir = compiler.attrib.get("meshdir")
    if meshdir and not os.path.isabs(meshdir):
        compiler.attrib["meshdir"] = os.path.abspath(os.path.join(os.path.dirname(xml_path), meshdir))


def load_mujoco_model(xml_path: str, simulation_dt: float, remove_bodies: Iterable[str] = ()) -> mujoco.MjModel:
    """Load a MuJoCo model, optionally removing named world bodies before compilation."""
    xml_path = resolve_repo_path(xml_path)
    remove_bodies = tuple(remove_bodies)

    if remove_bodies:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        _remove_named_bodies(root, set(remove_bodies))
        _make_meshdir_absolute(root, xml_path)
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    else:
        model = mujoco.MjModel.from_xml_path(xml_path)

    model.opt.timestep = simulation_dt
    return model


def make_policy_obs(frame_stack, qj, dqj, quat, omega, cmd, previous_action, config: Config) -> np.ndarray:
    default_angles = config.default_angles[config.policy_to_robot]
    qj_normalized = (qj - default_angles) * config.dof_pos_scale
    dqj_normalized = dqj * config.dof_vel_scale
    gravity_orientation = get_gravity_orientation(quat)
    omega_normalized = omega * config.ang_vel_scale

    return process_observation_with_frame_stack(
        frame_stack=frame_stack,
        omega=omega_normalized,
        gravity_orientation=gravity_orientation,
        cmd=cmd,
        qj=qj_normalized[config.robot_to_policy],
        dqj=dqj_normalized[config.robot_to_policy],
        action=previous_action[config.robot_to_policy],
        cmd_scale=config.cmd_scale,
        num_obs=config.num_obs,
        num_actions=config.num_actions,
    )


def _body_xmat(data: mujoco.MjData, model: mujoco.MjModel, body_name: str) -> np.ndarray:
    return data.xmat[model.body(body_name).id].reshape(3, 3)


def _body_xpos(data: mujoco.MjData, model: mujoco.MjModel, body_name: str) -> np.ndarray:
    return data.xpos[model.body(body_name).id].astype(np.float32)


def _body_cvel(data: mujoco.MjData, model: mujoco.MjModel, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    cvel = data.cvel[model.body(body_name).id].astype(np.float32)
    return cvel[:3], cvel[3:6]


def _projected_gravity_from_body(data: mujoco.MjData, model: mujoco.MjModel, body_name: str) -> np.ndarray:
    body_rot_world = _body_xmat(data, model, body_name)
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return (body_rot_world.T @ gravity_world).astype(np.float32)


def _relative_pos(data: mujoco.MjData, model: mujoco.MjModel, source_body: str, target_body: str) -> np.ndarray:
    source_pos = _body_xpos(data, model, source_body)
    target_pos = _body_xpos(data, model, target_body)
    source_rot = _body_xmat(data, model, source_body)
    return (source_rot.T @ (target_pos - source_pos)).astype(np.float32)


def build_stage3_teacher_encoder_obs(
    data: mujoco.MjData,
    model: mujoco.MjModel,
    qj: np.ndarray,
    dqj: np.ndarray,
    quat: np.ndarray,
    omega: np.ndarray,
    cmd: np.ndarray,
    previous_action: np.ndarray,
    config: Config,
) -> np.ndarray:
    """Build one Stage 3 teacher/adapter encoder observation timestep.

    Matches ObjectObservationsCfg.EncoderCfg:
    proprio(96) + base_lin_vel(3) + tray gravity(3) + tray pos in torso(3)
    + object pos in tray(3) + object angular velocity rel torso(3) * 0.2
    + object linear velocity rel torso(3) * 0.5 + object projected gravity(3).
    """
    default_angles = config.default_angles[config.policy_to_robot]
    qj_normalized = (qj - default_angles) * config.dof_pos_scale
    dqj_normalized = dqj * config.dof_vel_scale
    gravity_orientation = get_gravity_orientation(quat)
    omega_normalized = omega * config.ang_vel_scale

    obs_parts = [
        omega_normalized,
        gravity_orientation,
        cmd * config.cmd_scale,
        qj_normalized[config.robot_to_policy],
        dqj_normalized[config.robot_to_policy],
        previous_action[config.robot_to_policy],
    ]

    base_lin_vel = data.qvel[:3].astype(np.float32)
    tray_gravity = _projected_gravity_from_body(data, model, "plate")
    tray_pos_rel = _relative_pos(data, model, "torso_link", "plate")
    object_pos_rel = _relative_pos(data, model, "plate", "object")
    object_ang_vel, object_lin_vel = _body_cvel(data, model, "object")
    torso_ang_vel, torso_lin_vel = _body_cvel(data, model, "torso_link")
    object_gravity = _projected_gravity_from_body(data, model, "object")

    obs_parts.extend(
        [
            base_lin_vel,
            tray_gravity,
            np.clip(tray_pos_rel, -1.0, 1.0),
            np.clip(object_pos_rel, -1.0, 1.0),
            np.clip(object_ang_vel - torso_ang_vel, -50.0, 50.0) * 0.2,
            np.clip(object_lin_vel - torso_lin_vel, -10.0, 10.0) * 0.5,
            object_gravity,
        ]
    )
    return np.concatenate(obs_parts).astype(np.float32)


def initialize_model_state(model: mujoco.MjModel, data: mujoco.MjData, config: Config) -> np.ndarray:
    default_angles = config.default_angles[config.policy_to_robot]
    data.qpos[7 : 7 + config.num_actions] = default_angles
    mujoco.mj_forward(model, data)
    return default_angles.copy()


def configure_viewer_camera(viewer):
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = 1
    viewer.cam.distance = 2.5
    viewer.cam.elevation = -20
    viewer.cam.azimuth = 90
    viewer.cam.lookat[:] = [0, 0, 0.5]


def run_standard_stage(
    policy_path: str,
    config: Config,
    cmd: np.ndarray,
    duration: float | None = None,
    dry_run: bool = False,
) -> None:
    """Run a standard single-input Stage 1/2 MuJoCo deployment."""
    policy_path = resolve_repo_path(policy_path)
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy file not found: {policy_path}")

    model = load_mujoco_model(config.xml_path, config.simulation_dt, config.remove_bodies)
    data = mujoco.MjData(model)
    target_dof_pos = initialize_model_state(model, data, config)
    action = np.zeros(config.num_actions, dtype=np.float32)

    policy = torch.jit.load(policy_path).eval()
    frame_stack = deque(maxlen=5)
    for _ in range(5):
        frame_stack.append(np.zeros(config.num_obs, dtype=np.float32))

    def compute_once():
        nonlocal action, target_dof_pos
        qj = data.qpos[7 : 7 + config.num_actions].astype(np.float32)
        dqj = data.qvel[6 : 6 + config.num_actions].astype(np.float32)
        quat = data.qpos[3:7].astype(np.float32)
        omega = data.qvel[3:6].astype(np.float32)
        policy_obs = make_policy_obs(frame_stack, qj, dqj, quat, omega, cmd, action, config)
        with torch.no_grad():
            action_policy = policy(torch.from_numpy(policy_obs).unsqueeze(0)).squeeze(0).numpy()
        if action_policy.shape[0] != config.num_actions:
            raise ValueError(f"Policy returned {action_policy.shape[0]} actions, expected {config.num_actions}")
        action = action_policy[config.policy_to_robot]
        target_dof_pos = action * config.action_scale[config.policy_to_robot] + config.default_angles[config.policy_to_robot]

    if dry_run:
        compute_once()
        print(f"Dry run OK: policy={policy_path}, xml={config.xml_path}, action_dim={action.shape[0]}")
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
