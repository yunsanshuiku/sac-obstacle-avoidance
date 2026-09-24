"""Seeded evaluation with explicit outcomes and actual step counts.

Historical training scripts use different score/step conventions. This module
reports the same metrics for all scenarios without relabeling historical logs.
"""
import csv
import hashlib
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .policies import load_policy


def run_episode(policy, seed, max_steps):
    """Simulate one independently seeded scenario and return metrics and geometry."""
    scenario = policy.scenario
    if scenario == "static2d":
        from .static2d.fast import FastPathAvoidanceEnv
        env = FastPathAvoidanceEnv(seed=seed, max_steps=max_steps)
        obs = env.reset(randomize=True)
        raw = env.state.copy()
        static = env.obstacles
        dynamic = []
        dims = 2
    elif scenario == "dynamic2d":
        from .dynamic2d import SACAvoidanceEnv2D
        env = SACAvoidanceEnv2D(seed=seed)
        obs = env.reset(randomize_static=True)
        raw = env.agent_state.copy()
        static, dynamic = env.static_obstacles, env.dynamic_obstacles
        dims = 2
    else:
        from .env3d import AvoidanceEnv3D
        env = AvoidanceEnv3D(seed=seed)
        # Preserve the source experiment's obstacle initialization order.
        env._reset_dynamic_obstacles()
        env._randomize_static_obstacles()
        raw = env.start.copy()
        obs = env._state_features(raw)
        static, dynamic = env.static_obstacles, env.dynamic_obstacles
        dims = 3
    path = [raw.copy()]
    obstacle_paths = [[o.center.copy()] for o in dynamic]
    actions = []
    min_clearance = min(float(np.linalg.norm(raw[:dims] - o.center) - o.radius) for o in static + dynamic)
    success = collision = outside = False
    reward_sum = 0.0
    for step in range(max_steps):
        action = policy.act(obs)
        if scenario == "static2d":
            obs, reward, done, info = env.step(action)
            raw = env.state.copy()
            success, collision = bool(info["success"]), bool(info["collision"])
        elif scenario == "dynamic2d":
            obs, reward, done, info = env.step(action)
            raw = env.agent_state.copy()
            success, collision, outside = bool(info["reached"]), bool(info["collided"]), bool(info["outside"])
        else:
            next_raw = env.step(raw, action)
            success = bool(np.linalg.norm(next_raw[:3] - env.goal) < 0.40 and np.linalg.norm(next_raw[3:]) < 0.50)
            collision = bool(env.obstacle_penalty(next_raw[:3]) >= 1000.0)
            reward = 400.0 if success else (-200.0 if collision else env.shaped_reward(raw, action, next_raw))
            raw = next_raw
            obs = env._state_features(raw)
            done = success or collision
        if not np.isfinite(raw).all() or not np.isfinite(reward):
            raise FloatingPointError("Non-finite simulation state or reward.")
        reward_sum += float(reward)
        min_clearance = min(min_clearance, *(float(np.linalg.norm(raw[:dims] - o.center) - o.radius) for o in static + dynamic))
        path.append(raw.copy())
        actions.append(action.copy())
        for positions, obstacle in zip(obstacle_paths, dynamic):
            positions.append(obstacle.center.copy())
        if done:
            break
    path = np.asarray(path)
    outcome = "collision" if collision else ("success" if success else ("outside" if outside else "timeout"))
    metrics = {
        "seed": seed, "outcome": outcome, "success": outcome == "success",
        "collision": collision, "outside": outside, "timeout": outcome == "timeout",
        "steps": step + 1, "reward": reward_sum,
        "final_distance": float(np.linalg.norm(raw[:dims] - env.goal)),
        "min_clearance": min_clearance,
        "path_length": float(np.linalg.norm(np.diff(path[:, :dims], axis=0), axis=1).sum()),
    }
    return metrics, (env, path, np.asarray(actions), obstacle_paths, dims)


def plot_episode(geometry, metrics, output):
    """Plot the first sampled rollout, including failures rather than selecting a success."""
    env, path, actions, obstacle_paths, dims = geometry
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d" if dims == 3 else None)
    static = env.obstacles if hasattr(env, "obstacles") else env.static_obstacles
    dynamic = getattr(env, "dynamic_obstacles", [])
    if dims == 2:
        for obstacle in static:
            ax.add_patch(plt.Circle(obstacle.center, obstacle.radius, color="gray", alpha=0.4))
        for obstacle, positions in zip(dynamic, obstacle_paths):
            tr = np.asarray(positions)
            ax.plot(tr[:, 0], tr[:, 1], "--", alpha=0.6)
            ax.add_patch(plt.Circle(tr[0], obstacle.radius, color="orange", alpha=0.3))
        ax.plot(path[:, 0], path[:, 1], color="#126782", lw=2, label="SAC policy")
        ax.scatter(*env.goal, marker="*", s=150, color="#c44e52", label="Goal")
        ax.scatter(*path[0, :2], s=50, color="green", label="Start")
        ax.set_aspect("equal", adjustable="datalim")
    else:
        from .env3d import sphere_surface
        for obstacle in static:
            ax.plot_surface(*sphere_surface(obstacle.center, obstacle.radius, 10), color="gray", alpha=0.25)
        for obstacle, positions in zip(dynamic, obstacle_paths):
            tr = np.asarray(positions)
            ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], "--", alpha=0.6)
            ax.plot_surface(*sphere_surface(tr[0], obstacle.radius, 8), color="orange", alpha=0.25)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color="#126782", lw=2, label="SAC policy")
        ax.scatter(*env.goal, marker="*", s=150, color="#c44e52", label="Goal")
        ax.scatter(*path[0, :3], s=50, color="green", label="Start")
        ax.set_zlabel("z")
        ax.set_box_aspect((1, 1, 1))
    ax.set(xlabel="x", ylabel="y", title=f"Seed {metrics['seed']} | {metrics['outcome']} | {metrics['steps']} steps")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "trajectory.png", dpi=150)
    plt.close(fig)
    arrays = {"state": path, "action": actions, "goal": env.goal, "dt": np.array(env.dt)}
    arrays.update({f"dynamic_obstacle_{i}": np.asarray(tr) for i, tr in enumerate(obstacle_paths)})
    arrays["static_centers"] = np.asarray([o.center for o in static])
    arrays["static_radii"] = np.asarray([o.radius for o in static])
    arrays["dynamic_radii"] = np.asarray([o.radius for o in dynamic])
    np.savez_compressed(output / "rollout.npz", **arrays)


def evaluate(scenario, output, checkpoint=None, episodes=10, max_steps=600, seed=10000, device="cpu"):
    """Evaluate seeds [seed, seed + episodes), recording each outcome and provenance."""
    if episodes < 1 or max_steps < 1:
        raise ValueError("episodes and max_steps must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    policy = load_policy(scenario, checkpoint, device)
    rows = []
    for index in range(episodes):
        metrics, geometry = run_episode(policy, seed + index, max_steps)
        rows.append(metrics)
        if index == 0:
            plot_episode(geometry, metrics, output)
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "scenario": scenario, "episodes": episodes, "seed_start": seed, "max_steps": max_steps,
        "checkpoint_sha256": hashlib.sha256(policy.source.read_bytes()).hexdigest(),
        "python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
        "device": str(policy.device),
    }
    for key in ["success", "collision", "outside", "timeout"]:
        summary[f"{key}_rate"] = float(np.mean([r[key] for r in rows]))
    for key in ["steps", "reward", "final_distance", "min_clearance", "path_length"]:
        summary[f"mean_{key}"] = float(np.mean([r[key] for r in rows]))
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    return summary
