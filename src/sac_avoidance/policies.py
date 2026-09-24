"""Portable inference-only checkpoints shared by all three SAC experiments.

Each exported file contains tensors and basic Python metadata. Inference uses
the squashed Gaussian mean and does not allocate critics or a replay buffer.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

DIMENSIONS = {"static2d": (13, 2), "dynamic2d": (18, 2), "dynamic3d": (41, 3)}


def resolve_device(name="auto"):
    """Choose a device, failing explicitly when CUDA was requested but unavailable."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu or install a CUDA PyTorch build.")
    return torch.device(name)


def make_actor(scenario, hidden_dim=256):
    """Retain original parameter names and network shapes for checkpoint fidelity."""
    state_dim, action_dim = DIMENSIONS[scenario]
    if scenario == "static2d":
        from .static2d.core import GaussianPolicy
        if hidden_dim != 256:
            raise ValueError("The static2d architecture requires hidden_dim=256.")
        return GaussianPolicy(state_dim, action_dim, 1.2)
    if scenario == "dynamic2d":
        from .dynamic2d import GaussianPolicy
    else:
        from .dynamic3d import GaussianPolicy
    return GaussianPolicy(state_dim, action_dim, 1.2, hidden_dim)


@dataclass
class Policy:
    """A deterministic policy mapping one normalized observation to acceleration."""
    actor: torch.nn.Module
    scenario: str
    device: torch.device
    source: Path

    @torch.inference_mode()
    def act(self, observation, deterministic=True):
        obs = np.asarray(observation, dtype=np.float32)
        expected = (DIMENSIONS[self.scenario][0],)
        if obs.shape != expected or not np.isfinite(obs).all():
            raise ValueError(f"Expected a finite observation with shape {expected}, got {obs.shape}.")
        state = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        if deterministic:
            mean, _ = self.actor(state)
            action = torch.tanh(mean) * 1.2
        else:
            action, _, _ = self.actor.sample(state)
        result = action[0].cpu().numpy()
        if not np.isfinite(result).all():
            raise FloatingPointError("The policy produced a non-finite acceleration.")
        return result


def export_policy(actor, scenario, path, hidden_dim=256):
    """Save CPU actor tensors without training state or custom pickle objects."""
    state_dim, action_dim = DIMENSIONS[scenario]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": 1, "scenario": scenario, "state_dim": state_dim,
        "action_dim": action_dim, "hidden_dim": hidden_dim, "max_action": 1.2,
        "actor": {k: v.detach().cpu() for k, v in actor.state_dict().items()},
    }, path)


def load_policy(scenario, path=None, device="cpu"):
    """Load a bundled or exported policy; reject a different observation space."""
    path = Path(path) if path is not None else Path(__file__).parent / "pretrained" / f"{scenario}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Policy checkpoint not found: {path}")
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("format_version") != 1 or data.get("scenario") != scenario:
        raise ValueError(f"Expected a portable {scenario} policy. Use the exported policy.pt file.")
    if (data.get("state_dim"), data.get("action_dim")) != DIMENSIONS[scenario] or data.get("max_action") != 1.2:
        raise ValueError("Checkpoint dimensions or action scale do not match the environment.")
    actor = make_actor(scenario, data["hidden_dim"])
    actor.load_state_dict(data["actor"], strict=True)
    device = resolve_device(device)
    actor.to(device).eval()
    return Policy(actor, scenario, device, path)
