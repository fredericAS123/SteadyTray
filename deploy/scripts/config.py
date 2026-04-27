"""Shared configuration loader for both MuJoCo and real robot deployment."""

import os
import numpy as np
import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge two YAML dictionaries."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_with_base(file_path: str) -> dict:
    """Load a YAML file, optionally merging a base_config first."""
    with open(file_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if config is None:
        return {}

    base_config = config.pop("base_config", None)
    if base_config is None:
        return config

    if not os.path.isabs(base_config):
        base_config = os.path.join(os.path.dirname(file_path), base_config)

    return _deep_merge(_load_yaml_with_base(base_config), config)


class Config:
    """
    Configuration class that loads parameters from YAML files.
    
    This class is used by both MuJoCo simulation and real robot deployment
    to ensure consistent configuration across different deployment modes.
    """
    
    def __init__(self, file_path: str) -> None:
        """
        Load configuration from a YAML file.
        
        Args:
            file_path: Path to the YAML configuration file
        """
        config = _load_yaml_with_base(file_path)

        # Control parameters
        self.control_dt = config["control_dt"]
            
        # Robot-specific parameters (optional, for real robot)
        self.weak_motor = []
        if "weak_motor" in config:
            self.weak_motor = config["weak_motor"]

        # DDS communication topics (optional, for real robot)
        if "lowcmd_topic" in config:
            self.lowcmd_topic = config["lowcmd_topic"]
        if "lowstate_topic" in config:
            self.lowstate_topic = config["lowstate_topic"]

        # Joint mapping and PD gains
        self.policy_to_robot = config["policy_to_xml"]  # Maps policy indices to robot/xml motor indices
        self.robot_to_policy = config["xml_to_policy"]  # Maps robot/xml motor indices to policy indices
        self.kps = np.array(config["kps"], dtype=np.float32)
        self.kds = np.array(config["kds"], dtype=np.float32)
        self.default_angles = np.array(config["default_angles"], dtype=np.float32)

        # Observation scaling factors
        self.ang_vel_scale = config["ang_vel_scale"]
        self.dof_pos_scale = config["dof_pos_scale"]
        self.dof_vel_scale = config["dof_vel_scale"]
        self.action_scale = np.array(config["action_scale"], dtype=np.float32)
        self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        # Dimensions
        self.num_actions = config["num_actions"]
        self.num_obs = config["num_obs"]

        # Joint limits
        self.joint_limits_lower = np.array(config["joint_limits_lower"], dtype=np.float32)
        self.joint_limits_upper = np.array(config["joint_limits_upper"], dtype=np.float32)

        # Command limits (optional, for real robot joystick)
        if "vel_x_cmd" in config:
            self.vel_x_cmd = config["vel_x_cmd"]
        if "vel_y_cmd" in config:
            self.vel_y_cmd = config["vel_y_cmd"]
        if "yaw_cmd" in config:
            self.yaw_cmd = config["yaw_cmd"]
            
        # MuJoCo-specific parameters (optional)
        if "xml_path" in config:
            self.xml_path = config["xml_path"]
        if "simulation_duration" in config:
            self.simulation_duration = config["simulation_duration"]
        if "simulation_dt" in config:
            self.simulation_dt = config["simulation_dt"]
        if "control_decimation" in config:
            self.control_decimation = config["control_decimation"]
        if "policy_joints" in config:
            self.policy_joints = config["policy_joints"]
        if "cmd_init" in config:
            self.cmd_init = np.array(config["cmd_init"], dtype=np.float32)
        if "remove_bodies" in config:
            self.remove_bodies = list(config["remove_bodies"])
        else:
            self.remove_bodies = []
        if "stage3_teacher_encoder_obs_dim" in config:
            self.stage3_teacher_encoder_obs_dim = int(config["stage3_teacher_encoder_obs_dim"])
            
        # IMU configuration (optional, for real robot)
        if "imu_type" in config:
            self.imu_type = config["imu_type"]
        else:
            self.imu_type = "pelvis"  # default
    
    def __repr__(self) -> str:
        """String representation of the config."""
        return f"Config(num_actions={self.num_actions}, num_obs={self.num_obs}, control_dt={self.control_dt})"
