"""
SAC (Soft Actor-Critic) training for the 2D obstacle-avoidance task.

This version uses PyTorch library components instead of the hand-written MLP update.
No Gym or Stable-Baselines dependency is required.

Outputs are written to sac_training_run/:
  - sac_training_log.csv / .npz
  - sac_best_actor.pt / sac_final_actor.pt
  - sac_training_curves.png
  - sac_eval_curves.png
  - sac_best_trajectory_dense.png
  - sac_best_rollout_diagnostics.png
"""

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


PROJECT_DIR = Path(__file__).resolve().parent
OUT_DIR = Path("runs/static2d")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class CircleObstacle:
    center: np.ndarray
    radius: float

    def __init__(self, center, radius):
        self.center = np.array(center, dtype=np.float32)
        self.radius = float(radius)


class AvoidanceEnv:
    def __init__(self, seed=0, max_steps=360):
        self.rng = np.random.default_rng(seed)
        self.start = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.goal = np.array([8.0, 8.0], dtype=np.float32)
        self.obstacles = [
            CircleObstacle((3.0, 4.0), 1.0),
            CircleObstacle((5.5, 5.0), 1.0),
            CircleObstacle((6.5, 7.0), 0.8),
        ]
        self.dt = 0.05
        self.max_acc = 1.2
        self.max_speed = 2.0
        self.safe_margin = 0.55
        self.max_steps = max_steps
        self.state_scale = max(np.linalg.norm(self.goal - self.start[:2]), 1.0)
        self.vel_scale = self.max_acc * 2.0
        self.obs_dim = 4 + 3 * len(self.obstacles)
        self.act_dim = 2
        self.state = None
        self.step_count = 0

    def features(self, s):
        p = s[:2]
        v = s[2:]
        pn = (p - self.goal) / self.state_scale
        vn = v / self.vel_scale
        feat = [pn[0], pn[1], vn[0], vn[1]]
        for obs in self.obstacles:
            d = np.linalg.norm(p - obs.center) - obs.radius
            nd = np.clip(d / self.state_scale, -1.0, 1.0)
            direction = (obs.center - p) / (self.state_scale + 1e-8)
            feat.extend([nd, direction[0], direction[1]])
        return np.array(feat, dtype=np.float32)

    def reset(self, randomize=True):
        self.state = self.start.copy()
        if randomize:
            self.state += np.array([0.08, 0.08, 0.0, 0.0], dtype=np.float32) * self.rng.normal(size=4)
        self.step_count = 0
        return self.features(self.state)

    def obstacle_clearance(self, p):
        return min(np.linalg.norm(p - obs.center) - obs.radius for obs in self.obstacles)

    def obstacle_penalty(self, p):
        pen = 0.0
        for obs in self.obstacles:
            d = np.linalg.norm(p - obs.center) - obs.radius
            if d < 0.0:
                return 1000.0
            if d < self.safe_margin:
                pen += (self.safe_margin - d) ** 2 * 100.0
        return pen

    def dynamics(self, s, u):
        x, y, vx, vy = s
        ax, ay = np.clip(u, -self.max_acc, self.max_acc)
        vx2 = np.clip(vx + ax * self.dt, -self.max_speed, self.max_speed)
        vy2 = np.clip(vy + ay * self.dt, -self.max_speed, self.max_speed)
        x2 = x + vx * self.dt
        y2 = y + vy * self.dt
        return np.array([x2, y2, vx2, vy2], dtype=np.float32)

    def step(self, action):
        s = self.state
        u = np.clip(action, -self.max_acc, self.max_acc)
        s_next = self.dynamics(s, u)
        self.step_count += 1

        old_dist = np.linalg.norm(s[:2] - self.goal)
        new_dist = np.linalg.norm(s_next[:2] - self.goal)
        speed = np.linalg.norm(s_next[2:])
        collided = self.obstacle_penalty(s_next[:2]) >= 1000.0
        reached = (new_dist < 0.25 and speed < 0.35)
        timeout = self.step_count >= self.max_steps

        # SAC-friendly dense reward: progress + terminal success + safety + mild goal shaping.
        progress_reward = 25.0 * (old_dist - new_dist)
        distance_reward = -0.04 * new_dist
        control_cost = 0.002 * float(np.dot(u, u))
        obs_cost = min(self.obstacle_penalty(s_next[:2]), 80.0)
        near_goal_speed_cost = 0.0
        if new_dist < 1.0:
            near_goal_speed_cost = 1.5 * speed ** 2
        reward = progress_reward + distance_reward - control_cost - obs_cost - near_goal_speed_cost
        if reached:
            reward += 300.0
        if collided:
            reward -= 300.0
        if timeout and not reached:
            reward -= 20.0

        self.state = s_next
        done = reached or collided or timeout
        info = {
            "success": reached,
            "collision": collided,
            "timeout": timeout,
            "distance": float(new_dist),
            "speed": float(speed),
            "clearance": float(self.obstacle_clearance(s_next[:2])),
        }
        return self.features(s_next), float(reward), done, info


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, capacity=200_000):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, obs, act, rew, next_obs, done):
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx], device=DEVICE),
            torch.as_tensor(self.act[idx], device=DEVICE),
            torch.as_tensor(self.rew[idx], device=DEVICE),
            torch.as_tensor(self.next_obs[idx], device=DEVICE),
            torch.as_tensor(self.done[idx], device=DEVICE),
        )


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256, 256)):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, max_action):
        super().__init__()
        self.backbone = MLP(obs_dim, 256, hidden=(256, 256))
        self.mean = nn.Linear(256, act_dim)
        self.log_std = nn.Linear(256, act_dim)
        self.max_action = float(max_action)

    def forward(self, obs):
        h = self.backbone(obs)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), -5.0, 2.0)
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.rsample()
        tanh_z = torch.tanh(z)
        action = tanh_z * self.max_action
        log_prob = normal.log_prob(z) - torch.log(self.max_action * (1 - tanh_z.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        deterministic = torch.tanh(mean) * self.max_action
        return action, log_prob, deterministic

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        if deterministic:
            mean, _ = self(obs_t)
            action = torch.tanh(mean) * self.max_action
        else:
            action, _, _ = self.sample(obs_t)
        return action.cpu().numpy()[0]


class SACAgent:
    def __init__(self, obs_dim, act_dim, max_action):
        self.actor = GaussianPolicy(obs_dim, act_dim, max_action).to(DEVICE)
        self.q1 = MLP(obs_dim + act_dim, 1).to(DEVICE)
        self.q2 = MLP(obs_dim + act_dim, 1).to(DEVICE)
        self.q1_target = MLP(obs_dim + act_dim, 1).to(DEVICE)
        self.q2_target = MLP(obs_dim + act_dim, 1).to(DEVICE)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=3e-4)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=3e-4)

        self.log_alpha = torch.tensor(math.log(0.2), dtype=torch.float32, device=DEVICE, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self.target_entropy = -float(act_dim)
        self.gamma = 0.99
        self.tau = 0.005

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def update(self, replay, batch_size=256):
        obs, act, rew, next_obs, done = replay.sample(batch_size)

        with torch.no_grad():
            next_act, next_logp, _ = self.actor.sample(next_obs)
            next_input = torch.cat([next_obs, next_act], dim=-1)
            target_q = torch.min(self.q1_target(next_input), self.q2_target(next_input))
            target = rew + self.gamma * (1.0 - done) * (target_q - self.alpha * next_logp)

        q_input = torch.cat([obs, act], dim=-1)
        q1_loss = F.mse_loss(self.q1(q_input), target)
        q2_loss = F.mse_loss(self.q2(q_input), target)
        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()
        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        new_act, logp, _ = self.actor.sample(obs)
        new_input = torch.cat([obs, new_act], dim=-1)
        q_new = torch.min(self.q1(new_input), self.q2(new_input))
        actor_loss = (self.alpha.detach() * logp - q_new).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for p, tp in zip(self.q1.parameters(), self.q1_target.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
            for p, tp in zip(self.q2.parameters(), self.q2_target.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.item()),
        }


def evaluate(agent, episodes=20, seed=10000):
    env = AvoidanceEnv(seed=seed, max_steps=500)
    rewards, successes, collisions, steps, distances, clearances = [], [], [], [], [], []
    for ep in range(episodes):
        obs = env.reset(randomize=True)
        ep_reward = 0.0
        min_clear = np.inf
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
    return {
        "eval_reward_mean": float(np.mean(rewards)),
        "eval_success_rate": float(np.mean(successes)),
        "eval_collision_rate": float(np.mean(collisions)),
        "eval_steps_mean": float(np.mean(steps)),
        "eval_final_distance_mean": float(np.mean(distances)),
        "eval_min_clearance_mean": float(np.mean(clearances)),
    }


def moving_average(x, window):
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return x
    window = min(window, len(x))
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    return np.convolve(np.pad(x, (pad_left, pad_right), mode="edge"), np.ones(window) / window, mode="valid")


def plot_curves(logs):
    ep = np.array([r["episode"] for r in logs])
    reward = np.array([r["train_reward"] for r in logs])
    success = np.array([r["train_success"] for r in logs])
    collision = np.array([r["train_collision"] for r in logs])
    final_dist = np.array([r["train_final_distance"] for r in logs])
    alpha = np.array([r.get("alpha", np.nan) for r in logs])

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].plot(ep, reward, color="0.75", linewidth=0.7, label="episode reward")
    axes[0, 0].plot(ep, moving_average(reward, 20), "b", linewidth=2, label="MA-20")
    axes[0, 0].plot(ep, moving_average(reward, 80), "r", linewidth=2, label="MA-80")
    axes[0, 0].set_title("SAC training reward convergence")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(ep, moving_average(success, 20) * 100, "g", linewidth=2, label="success MA-20")
    axes[0, 1].plot(ep, moving_average(collision, 20) * 100, "r", linewidth=2, label="collision MA-20")
    axes[0, 1].set_title("SAC training success/collision rates")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Rate (%)")
    axes[0, 1].set_ylim(-5, 105)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(ep, final_dist, color="0.75", linewidth=0.7)
    axes[1, 0].plot(ep, moving_average(final_dist, 50), "b", linewidth=2)
    axes[1, 0].set_title("Training final distance to goal")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Distance")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(ep, alpha, "m", linewidth=1.5)
    axes[1, 1].set_title("SAC entropy temperature alpha")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("alpha")
    axes[1, 1].grid(True, alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "sac_training_curves.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)

    eval_logs = [r for r in logs if "eval_success_rate" in r]
    ep2 = np.array([r["episode"] for r in eval_logs])
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].plot(ep2, [r["eval_reward_mean"] for r in eval_logs], "bo-", linewidth=2, markersize=4)
    axes[0, 0].set_title("SAC evaluation reward")
    axes[0, 1].plot(ep2, np.array([r["eval_success_rate"] for r in eval_logs]) * 100, "go-", label="success")
    axes[0, 1].plot(ep2, np.array([r["eval_collision_rate"] for r in eval_logs]) * 100, "ro-", label="collision")
    axes[0, 1].set_ylim(-5, 105)
    axes[0, 1].set_title("SAC evaluation success/collision rates")
    axes[0, 1].legend()
    axes[1, 0].plot(ep2, [r["eval_final_distance_mean"] for r in eval_logs], "bo-")
    axes[1, 0].set_title("SAC evaluation final distance")
    axes[1, 1].plot(ep2, [r["eval_min_clearance_mean"] for r in eval_logs], "mo-")
    axes[1, 1].axhline(0.0, color="r", linestyle="--", label="collision boundary")
    axes[1, 1].set_title("SAC evaluation minimum clearance")
    axes[1, 1].legend()
    for ax in axes.ravel():
        ax.set_xlabel("Episode")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    eval_out = OUT_DIR / "sac_eval_curves.png"
    fig.savefig(eval_out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out, eval_out


def rollout_path(agent):
    env = AvoidanceEnv(seed=123, max_steps=500)
    obs = env.reset(randomize=False)
    states = [env.state.copy()]
    actions, rewards, clearances = [], [], []
    info = {"success": False, "collision": False, "distance": float(np.linalg.norm(env.state[:2] - env.goal)), "speed": 0.0}
    for _ in range(env.max_steps):
        action = agent.actor.act(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        states.append(env.state.copy())
        actions.append(float(np.linalg.norm(action)))
        rewards.append(reward)
        clearances.append(info["clearance"])
        if done:
            break
    return env, np.asarray(states), np.asarray(actions), np.asarray(rewards), np.asarray(clearances), info


def plot_rollout(agent):
    env, path, actions, rewards, clearances, info = rollout_path(agent)
    theta = np.linspace(0, 2 * np.pi, 240)

    fig, ax = plt.subplots(figsize=(9, 9))
    idx = np.arange(len(path))
    sc = ax.scatter(path[:, 0], path[:, 1], c=idx, cmap="viridis", s=18, alpha=0.95, label="every step")
    ax.plot(path[:, 0], path[:, 1], "b-", linewidth=1.3, alpha=0.7)
    mark_every = max(1, len(path) // 30)
    ax.plot(path[::mark_every, 0], path[::mark_every, 1], "wo", markersize=3, markeredgecolor="k", markeredgewidth=0.4,
            label=f"marker every {mark_every} steps")
    ax.scatter(env.start[0], env.start[1], c="g", s=130, label="start", zorder=5)
    ax.scatter(env.goal[0], env.goal[1], c="r", s=130, label="goal", zorder=5)
    for i, obs in enumerate(env.obstacles):
        ax.fill(obs.center[0] + obs.radius * np.cos(theta), obs.center[1] + obs.radius * np.sin(theta),
                color="gray", alpha=0.35, label="obstacle" if i == 0 else None)
        safe_r = obs.radius + env.safe_margin
        ax.plot(obs.center[0] + safe_r * np.cos(theta), obs.center[1] + safe_r * np.sin(theta),
                "k--", alpha=0.45, label="safe boundary" if i == 0 else None)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Step index")
    ax.set_title(f"SAC best trajectory | success={info['success']} collision={info['collision']}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    traj_out = OUT_DIR / "sac_best_trajectory_dense.png"
    fig.savefig(traj_out, dpi=240, bbox_inches="tight")
    plt.close(fig)

    t = np.arange(len(path)) * env.dt
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].plot(t, np.linalg.norm(path[:, :2] - env.goal, axis=1), linewidth=1.8)
    axes[0, 0].set_title("Distance to goal")
    axes[0, 1].plot(t, np.linalg.norm(path[:, 2:], axis=1), linewidth=1.8)
    axes[0, 1].set_title("Speed profile")
    axes[1, 0].plot(t[1:], clearances, linewidth=1.8)
    axes[1, 0].axhline(0.0, color="r", linestyle="--", label="collision boundary")
    axes[1, 0].axhline(env.safe_margin, color="k", linestyle="--", alpha=0.5, label="safe margin")
    axes[1, 0].set_title("Minimum obstacle clearance")
    axes[1, 0].legend()
    axes[1, 1].plot(t[1:], actions, label="action norm", linewidth=1.8)
    axes[1, 1].plot(t[1:], rewards, label="instant reward", linewidth=1.2, alpha=0.8)
    axes[1, 1].set_title("Control effort and reward")
    axes[1, 1].legend()
    for ax in axes.ravel():
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    diag_out = OUT_DIR / "sac_best_rollout_diagnostics.png"
    fig.savefig(diag_out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return traj_out, diag_out
