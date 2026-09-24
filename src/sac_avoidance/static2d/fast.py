"""
Fast-path SAC variant for the 2D obstacle-avoidance task.

Goal of this variant:
  - preserve SAC's successful obstacle avoidance behavior
  - prefer shorter arrival time
  - prefer shorter geometric path length

It reuses the PyTorch SAC implementation in core.py, while retaining the original
fast-path reward and best-model scoring convention.
"""

import csv
from pathlib import Path

import numpy as np
import torch

from . import core as sac


PROJECT_DIR = Path(__file__).resolve().parent
OUT_DIR = Path("runs/static2d")


class FastPathAvoidanceEnv(sac.AvoidanceEnv):
    """SAC environment with stronger incentives for short time and short path."""

    def reset(self, randomize=True):
        obs = super().reset(randomize=randomize)
        self.path_length = 0.0
        return obs

    def step(self, action):
        s = self.state
        u = np.clip(action, -self.max_acc, self.max_acc)
        s_next = self.dynamics(s, u)
        self.step_count += 1

        old_dist = np.linalg.norm(s[:2] - self.goal)
        new_dist = np.linalg.norm(s_next[:2] - self.goal)
        speed = np.linalg.norm(s_next[2:])
        step_length = float(np.linalg.norm(s_next[:2] - s[:2]))
        self.path_length += step_length

        collided = self.obstacle_penalty(s_next[:2]) >= 1000.0
        reached = (new_dist < 0.25 and speed < 0.35)
        timeout = self.step_count >= self.max_steps

        # Stronger progress reward encourages direct motion toward the goal.
        progress_reward = 35.0 * (old_dist - new_dist)

        # Small distance shaping avoids wandering far from the goal.
        distance_cost = 0.03 * new_dist

        # NEW: per-step time penalty. This directly encourages fewer steps / shorter time.
        time_cost = 0.18

        # NEW: geometric path-length penalty. Detours become more expensive.
        path_cost = 0.08 * step_length

        # Keep control penalty modest so the agent is allowed to move quickly.
        control_cost = 0.001 * float(np.dot(u, u))

        # Same safety idea as the original SAC run.
        obs_cost = min(self.obstacle_penalty(s_next[:2]), 80.0)

        # Only slow down near the target, because success requires low terminal speed.
        near_goal_speed_cost = 0.0
        if new_dist < 1.0:
            near_goal_speed_cost = 1.2 * speed ** 2

        # Mild speed reward far from the goal helps reduce travel time.
        speed_bonus = 0.04 * speed if new_dist > 1.5 else 0.0

        reward = (progress_reward + speed_bonus
                  - distance_cost - time_cost - path_cost
                  - control_cost - obs_cost - near_goal_speed_cost)

        if reached:
            # Earlier arrival receives a larger terminal bonus.
            fast_bonus = 160.0 * (1.0 - self.step_count / self.max_steps)
            reward += 320.0 + max(0.0, fast_bonus)
        if collided:
            reward -= 320.0
        if timeout and not reached:
            reward -= 40.0

        self.state = s_next
        done = reached or collided or timeout
        info = {
            "success": reached,
            "collision": collided,
            "timeout": timeout,
            "distance": float(new_dist),
            "speed": float(speed),
            "clearance": float(self.obstacle_clearance(s_next[:2])),
            "path_length": float(self.path_length),
        }
        return self.features(s_next), float(reward), done, info


def evaluate_fast(agent, episodes=20, seed=10000, max_steps=420):
    env = FastPathAvoidanceEnv(seed=seed, max_steps=max_steps)
    rewards, successes, collisions, steps, distances, clearances, path_lengths = [], [], [], [], [], [], []
    for ep in range(episodes):
        obs = env.reset(randomize=True)
        ep_reward = 0.0
        min_clear = np.inf
        info = {"success": False, "collision": False, "distance": np.nan, "path_length": np.nan}
        for step in range(env.max_steps):
            action = agent.actor.act(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            min_clear = min(min_clear, info["clearance"])
            if done:
                break
        rewards.append(ep_reward)
        successes.append(float(info["success"]))
        collisions.append(float(info["collision"]))
        steps.append(step + 1)
        distances.append(info["distance"])
        clearances.append(min_clear)
        path_lengths.append(info["path_length"])
    return {
        "eval_reward_mean": float(np.mean(rewards)),
        "eval_success_rate": float(np.mean(successes)),
        "eval_collision_rate": float(np.mean(collisions)),
        "eval_steps_mean": float(np.mean(steps)),
        "eval_final_distance_mean": float(np.mean(distances)),
        "eval_min_clearance_mean": float(np.mean(clearances)),
        "eval_path_length_mean": float(np.mean(path_lengths)),
    }
