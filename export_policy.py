#!/usr/bin/env python3
"""Export a Stage-1/Stage-2 RSL-RL checkpoint (.pt) to TorchScript policy (.jit/.pt).

Usage:
  python export_policy.py \
      --checkpoint logs/rsl_rl/.../model_5200.pt \
      --output logs/rsl_rl/.../policy.jit
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from rsl_rl.modules import ActorCritic


class ExportActorPolicy(nn.Module):
    """Wrapper that exposes a single-input forward(observations) -> actions."""

    def __init__(self, actor_critic: ActorCritic):
        super().__init__()
        self.actor_critic = actor_critic

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor_critic.act_inference(observations)


def _extract_state_dict(ckpt_obj: object) -> dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict):
        if "model_state_dict" in ckpt_obj and isinstance(ckpt_obj["model_state_dict"], dict):
            return ckpt_obj["model_state_dict"]
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
        # pure state_dict
        if ckpt_obj and all(isinstance(k, str) for k in ckpt_obj.keys()):
            return ckpt_obj  # type: ignore[return-value]
    raise ValueError("Unsupported checkpoint format. Expected a training dict with model_state_dict or a pure state_dict.")


def _infer_mlp_dims(state_dict: dict[str, torch.Tensor], prefix: str) -> tuple[int, list[int], int]:
    # Example keys: actor.0.weight, actor.2.weight, actor.4.weight, actor.6.weight
    weight_keys = [k for k in state_dict.keys() if k.startswith(f"{prefix}.") and k.endswith(".weight")]
    if not weight_keys:
        raise ValueError(f"No MLP weight keys found for prefix '{prefix}'.")

    def _layer_idx(k: str) -> int:
        return int(k.split(".")[1])

    sorted_keys = sorted(weight_keys, key=_layer_idx)
    in_dim = state_dict[sorted_keys[0]].shape[1]
    out_dim = state_dict[sorted_keys[-1]].shape[0]
    hidden_dims = [state_dict[k].shape[0] for k in sorted_keys[:-1]]
    return int(in_dim), [int(x) for x in hidden_dims], int(out_dim)


def export_policy(checkpoint: Path, output: Path, activation: str = "elu", init_noise_std: float = 1.0) -> Path:
    ckpt_obj = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(ckpt_obj)

    num_actor_obs, actor_hidden_dims, num_actions = _infer_mlp_dims(state_dict, "actor")
    num_critic_obs, critic_hidden_dims, _ = _infer_mlp_dims(state_dict, "critic")

    actor_critic = ActorCritic(
        num_actor_obs=num_actor_obs,
        num_critic_obs=num_critic_obs,
        num_actions=num_actions,
        actor_hidden_dims=actor_hidden_dims,
        critic_hidden_dims=critic_hidden_dims,
        activation=activation,
        init_noise_std=init_noise_std,
    )

    load_result = actor_critic.load_state_dict(state_dict, strict=False)
    # Different rsl_rl versions may return:
    # - bool (resumed_training)
    # - torch.nn.modules.module._IncompatibleKeys(missing_keys, unexpected_keys)
    # - tuple(missing_keys, unexpected_keys)
    if isinstance(load_result, tuple) and len(load_result) == 2:
        missing, unexpected = load_result
        if missing:
            print(f"[WARN] Missing keys when loading checkpoint: {len(missing)}")
        if unexpected:
            print(f"[WARN] Unexpected keys when loading checkpoint: {len(unexpected)}")
    elif hasattr(load_result, "missing_keys") and hasattr(load_result, "unexpected_keys"):
        if load_result.missing_keys:
            print(f"[WARN] Missing keys when loading checkpoint: {len(load_result.missing_keys)}")
        if load_result.unexpected_keys:
            print(f"[WARN] Unexpected keys when loading checkpoint: {len(load_result.unexpected_keys)}")

    actor_critic.eval()
    export_model = ExportActorPolicy(actor_critic).eval()

    example_obs = torch.zeros(1, num_actor_obs, dtype=torch.float32)
    with torch.no_grad():
        scripted = torch.jit.trace(export_model, example_obs)

    output.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(output))

    # quick runtime check
    loaded = torch.jit.load(str(output), map_location="cpu")
    with torch.no_grad():
        y = loaded(example_obs)
    print(f"[OK] Exported TorchScript policy: {output}")
    print(f"[INFO] Input dim: {num_actor_obs}, Output dim: {num_actions}, Check output shape: {tuple(y.shape)}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export RSL-RL checkpoint to TorchScript policy")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--output", type=Path, default=None, help="Output policy path (.jit/.pt)")
    parser.add_argument("--activation", type=str, default="elu", help="Policy activation used in training")
    parser.add_argument("--init-noise-std", type=float, default=1.0, help="Initial action noise std used in training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = args.checkpoint.expanduser().resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    if args.output is None:
        out = ckpt.parent / "policy.jit"
    else:
        out = args.output.expanduser().resolve()

    export_policy(
        checkpoint=ckpt,
        output=out,
        activation=args.activation,
        init_noise_std=args.init_noise_std,
    )


if __name__ == "__main__":
    main()
