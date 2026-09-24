"""
SAC 版二维动态避障训练脚本 —— 独立于 adpex_dynamic.py

输出全部放在 sac_outputs/，不覆盖旧 ADP 模型和旧图。

用法：
  python adpex_sac_dynamic.py
  python adpex_sac_dynamic.py --episodes 5 --steps-per-episode 50 --eval-every 2 --eval-scenarios 4 --warmup-steps 100
"""

from collections import deque
import json
import os
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from .checkpoints import basic_metadata


# =========================================================================
# 全局常量
# =========================================================================
NUM_SENSORS = 12
SENSOR_MAX_RANGE = 4.0
NUM_DYNAMIC_OBS = 3
DYNAMIC_OBS_SPEED_LO = 0.25
DYNAMIC_OBS_SPEED_HI = 0.55
ENV_XMIN, ENV_XMAX = -1.0, 10.0
ENV_YMIN, ENV_YMAX = -1.0, 10.0
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0
EPS = 1e-6


# =========================================================================
# 环境元素
# =========================================================================
class CircleObstacle:
    """圆形障碍物，由中心坐标和半径定义。"""

    def __init__(self, center, radius):
        self.center = np.array(center, dtype=float)
        self.radius = float(radius)

    def copy(self):
        return CircleObstacle(self.center.copy(), self.radius)


class DynamicObstacle:
    """具有恒定速度和边界反弹行为的动态圆形障碍物。"""

    def __init__(self, center, radius, velocity):
        self.center = np.array(center, dtype=float)
        self.radius = float(radius)
        self.velocity = np.array(velocity, dtype=float)
        self.trajectory = [self.center.copy()]

    def step(self, dt):
        self.center += self.velocity * dt
        r = self.radius
        if self.center[0] - r < ENV_XMIN:
            self.center[0] = ENV_XMIN + r
            self.velocity[0] *= -1.0
        elif self.center[0] + r > ENV_XMAX:
            self.center[0] = ENV_XMAX - r
            self.velocity[0] *= -1.0
        if self.center[1] - r < ENV_YMIN:
            self.center[1] = ENV_YMIN + r
            self.velocity[1] *= -1.0
        elif self.center[1] + r > ENV_YMAX:
            self.center[1] = ENV_YMAX - r
            self.velocity[1] *= -1.0
        self.trajectory.append(self.center.copy())

    def copy(self):
        return DynamicObstacle(self.center.copy(), self.radius, self.velocity.copy())


def clone_static_obstacles(obstacles):
    return [obs.copy() for obs in obstacles]


def clone_dynamic_obstacles(obstacles):
    return [obs.copy() for obs in obstacles]


# =========================================================================
# SAC 避障环境
# =========================================================================
class SACAvoidanceEnv2D:
    """二维连续动作避障环境，状态为 18 维，动作为二维加速度。"""

    def __init__(self, start=(0.0, 0.0, 0.0, 0.0), goal=(8.0, 8.0),
                 static_obstacles=None, dt=0.05, max_acc=1.2,
                 safe_margin=0.55, max_speed=2.0, seed=0):
        self.rng = np.random.default_rng(seed)
        self.start = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.dt = float(dt)
        self.max_acc = float(max_acc)
        self.safe_margin = float(safe_margin)
        self.max_speed = float(max_speed)
        self.state_scale = max(np.linalg.norm(self.goal - self.start[:2]), 1.0)

        if static_obstacles is None:
            static_obstacles = [
                CircleObstacle(center=(3.0, 4.0), radius=1.0),
                CircleObstacle(center=(5.5, 5.0), radius=1.0),
                CircleObstacle(center=(6.5, 7.0), radius=0.8),
            ]
        self.default_static_obstacles = clone_static_obstacles(static_obstacles)
        self.static_obstacles = clone_static_obstacles(static_obstacles)
        self.dynamic_obstacles = []
        self.agent_state = self.start.copy()
        self.last_path_length = 0.0
        self.episode_step = 0

        goal_vec = self.goal - self.start[:2]
        self.goal_distance = max(np.linalg.norm(goal_vec), 1e-6)
        self.goal_dir = goal_vec / self.goal_distance

        self.sensor_angles = np.linspace(0, 2 * np.pi, NUM_SENSORS, endpoint=False)
        self.sensor_dirs = np.column_stack([np.cos(self.sensor_angles), np.sin(self.sensor_angles)])
        self.state_dim = 2 + 2 + 2 + NUM_SENSORS
        self.action_dim = 2

        self._init_dynamic_obstacles()

    def _restore_default_static_obstacles(self):
        self.static_obstacles = clone_static_obstacles(self.default_static_obstacles)

    def _randomize_static_obstacles(self, num_static=3):
        new_obs = []
        for _ in range(num_static):
            for _ in range(300):
                center = np.array([
                    self.rng.uniform(ENV_XMIN + 1.0, ENV_XMAX - 1.0),
                    self.rng.uniform(ENV_YMIN + 1.0, ENV_YMAX - 1.0),
                ])
                radius = self.rng.uniform(0.5, 1.15)
                if np.linalg.norm(center - self.start[:2]) < 1.6:
                    continue
                if np.linalg.norm(center - self.goal) < 1.6:
                    continue
                overlap = False
                for obs in new_obs:
                    if np.linalg.norm(center - obs.center) < radius + obs.radius + 0.35:
                        overlap = True
                        break
                if not overlap:
                    new_obs.append(CircleObstacle(center, radius))
                    break
        self.static_obstacles = new_obs

    def _init_dynamic_obstacles(self):
        self.dynamic_obstacles = []
        for _ in range(NUM_DYNAMIC_OBS):
            for _ in range(300):
                center = np.array([
                    self.rng.uniform(ENV_XMIN + 1.5, ENV_XMAX - 1.5),
                    self.rng.uniform(ENV_YMIN + 1.5, ENV_YMAX - 1.5),
                ])
                radius = self.rng.uniform(0.35, 0.65)
                overlap = False
                for obs in self.static_obstacles:
                    if np.linalg.norm(center - obs.center) < radius + obs.radius + 0.55:
                        overlap = True
                        break
                if np.linalg.norm(center - self.start[:2]) < 1.6:
                    overlap = True
                if np.linalg.norm(center - self.goal) < 1.6:
                    overlap = True
                for dob in self.dynamic_obstacles:
                    if np.linalg.norm(center - dob.center) < radius + dob.radius + 0.35:
                        overlap = True
                        break
                if not overlap:
                    speed = self.rng.uniform(DYNAMIC_OBS_SPEED_LO, DYNAMIC_OBS_SPEED_HI)
                    angle = self.rng.uniform(0.0, 2 * np.pi)
                    velocity = np.array([speed * np.cos(angle), speed * np.sin(angle)])
                    self.dynamic_obstacles.append(DynamicObstacle(center, radius, velocity))
                    break

    def _all_obstacles(self):
        return self.static_obstacles + self.dynamic_obstacles

    def _sensor_readings(self, agent_pos):
        readings = np.full(NUM_SENSORS, SENSOR_MAX_RANGE, dtype=float)
        for i, d in enumerate(self.sensor_dirs):
            for obs in self._all_obstacles():
                v = agent_pos - obs.center
                b_half = np.dot(v, d)
                c_val = np.dot(v, v) - obs.radius ** 2
                disc = b_half ** 2 - c_val
                if disc < 0:
                    continue
                sqrt_disc = np.sqrt(disc)
                t1 = -b_half - sqrt_disc
                t2 = -b_half + sqrt_disc
                if t1 > 1e-6:
                    t = t1
                elif t2 > 1e-6:
                    t = t2
                else:
                    continue
                if t < readings[i]:
                    readings[i] = t
        return np.clip(readings, 0.0, SENSOR_MAX_RANGE)

    def state_features(self, agent_state=None):
        if agent_state is None:
            agent_state = self.agent_state
        p = agent_state[:2]
        v = agent_state[2:]
        rel_pos = (p - self.goal) / self.state_scale
        vel_norm = v / self.max_speed
        goal_norm = self.goal / self.state_scale
        sensors = self._sensor_readings(p) / SENSOR_MAX_RANGE
        return np.concatenate([rel_pos, vel_norm, goal_norm, sensors]).astype(np.float32)

    def _agent_dynamics(self, s, u):
        x, y, vx, vy = s
        ax, ay = np.clip(u, -self.max_acc, self.max_acc)
        vx2 = np.clip(vx + ax * self.dt, -self.max_speed, self.max_speed)
        vy2 = np.clip(vy + ay * self.dt, -self.max_speed, self.max_speed)
        x2 = x + vx * self.dt
        y2 = y + vy * self.dt
        return np.array([x2, y2, vx2, vy2], dtype=float)

    def obstacle_penalty(self, p):
        pen = 0.0
        for obs in self._all_obstacles():
            d = np.linalg.norm(p - obs.center) - obs.radius
            if d < 0.0:
                return 1000.0
            if d < self.safe_margin:
                pen += (self.safe_margin - d) ** 2 * 70.0
        return pen

    def min_obstacle_distance(self, p):
        min_d = SENSOR_MAX_RANGE
        for obs in self._all_obstacles():
            d = np.linalg.norm(p - obs.center) - obs.radius
            min_d = min(min_d, d)
        return min_d

    def shaped_reward(self, s, u, s_next):
        old_dist = np.linalg.norm(s[:2] - self.goal)
        new_dist = np.linalg.norm(s_next[:2] - self.goal)
        dist_reduction = old_dist - new_dist
        displacement = s_next[:2] - s[:2]
        forward_progress = np.dot(displacement, self.goal_dir)

        rel_from_start = s_next[:2] - self.start[:2]
        progress_along_line = np.dot(rel_from_start, self.goal_dir)
        closest_on_line = self.start[:2] + np.clip(
            progress_along_line, 0.0, self.goal_distance
        ) * self.goal_dir
        lateral_deviation = np.linalg.norm(s_next[:2] - closest_on_line)
        corridor_penalty = max(0.0, lateral_deviation - 2.0) ** 2

        control_cost = 0.002 * np.linalg.norm(u) ** 2
        step_cost = 0.025
        obs_cost = min(self.obstacle_penalty(s_next[:2]), 60.0)
        proximity_bonus = max(0.0, 2.0 - new_dist) * 0.8

        return (16.0 * dist_reduction
                + 7.0 * forward_progress
                - 0.12 * corridor_penalty
                - control_cost
                - step_cost
                - obs_cost
                + proximity_bonus)

    def reset(self, randomize_static=True, scenario=None):
        if scenario is not None:
            self.static_obstacles = clone_static_obstacles(scenario['static_obstacles'])
            self.dynamic_obstacles = clone_dynamic_obstacles(scenario['dynamic_obstacles'])
        else:
            if randomize_static:
                self._randomize_static_obstacles()
            else:
                self._restore_default_static_obstacles()
            self._init_dynamic_obstacles()
        self.agent_state = self.start.copy()
        self.agent_state += np.array([0.04, 0.04, 0.0, 0.0]) * self.rng.normal(size=4)
        self.last_path_length = 0.0
        self.episode_step = 0
        return self.state_features()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -self.max_acc, self.max_acc)
        s = self.agent_state.copy()
        s_next = self._agent_dynamics(s, action)
        for dob in self.dynamic_obstacles:
            dob.step(self.dt)
        self.agent_state = s_next
        self.episode_step += 1
        self.last_path_length += np.linalg.norm(s_next[:2] - s[:2])

        reached = (np.linalg.norm(s_next[:2] - self.goal) < 0.30
                   and np.linalg.norm(s_next[2:]) < 0.45)
        collided = self.obstacle_penalty(s_next[:2]) >= 1000.0
        outside = (s_next[0] < ENV_XMIN - 0.5 or s_next[0] > ENV_XMAX + 0.5 or
                   s_next[1] < ENV_YMIN - 0.5 or s_next[1] > ENV_YMAX + 0.5)

        if reached:
            reward = 220.0 - 0.35 * self.episode_step - 8.0 * max(0.0, self.last_path_length - self.goal_distance)
        elif collided:
            reward = -220.0
        elif outside:
            reward = -180.0
        else:
            reward = self.shaped_reward(s, action, s_next)

        done = reached or collided or outside
        info = {
            'reached': reached,
            'collided': collided,
            'outside': outside,
            'dist_to_goal': np.linalg.norm(s_next[:2] - self.goal),
            'path_length': self.last_path_length,
            'min_obs_dist': self.min_obstacle_distance(s_next[:2]),
        }
        return self.state_features(), float(reward), done, info

    def random_action(self):
        return self.rng.uniform(-self.max_acc, self.max_acc, size=2).astype(np.float32)

    def build_eval_scenarios(self, num_scenarios=30, seed=2026):
        old_rng = self.rng
        self.rng = np.random.default_rng(seed)
        scenarios = []
        for _ in range(num_scenarios):
            self._randomize_static_obstacles()
            self._init_dynamic_obstacles()
            scenarios.append({
                'static_obstacles': clone_static_obstacles(self.static_obstacles),
                'dynamic_obstacles': clone_dynamic_obstacles(self.dynamic_obstacles),
            })
        self.rng = old_rng
        self._restore_default_static_obstacles()
        self._init_dynamic_obstacles()
        return scenarios


# =========================================================================
# SAC 网络与经验回放
# =========================================================================
class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim, seed=0):
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, s, a, r, s_next, done):
        self.states[self.pos] = s
        self.actions[self.pos] = a
        self.rewards[self.pos] = r
        self.next_states[self.pos] = s_next
        self.dones[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.states[idx], device=device),
            torch.as_tensor(self.actions[idx], device=device),
            torch.as_tensor(self.rewards[idx], device=device),
            torch.as_tensor(self.next_states[idx], device=device),
            torch.as_tensor(self.dones[idx], device=device),
        )


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, action_scale, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.register_buffer('action_scale', torch.as_tensor(action_scale, dtype=torch.float32))

    def forward(self, state):
        h = self.net(state)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1.0 - y_t.pow(2)) + EPS)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        deterministic = torch.tanh(mean) * self.action_scale
        return action, log_prob, deterministic

    @torch.no_grad()
    def act(self, state_np, device, deterministic=False):
        state = torch.as_tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
        action, _, det = self.sample(state)
        out = det if deterministic else action
        return out.squeeze(0).cpu().numpy()


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


@dataclass
class SACConfig:
    episodes: int = 3000
    steps_per_episode: int = 500
    eval_every: int = 100
    eval_scenarios: int = 30
    save_every: int = 100
    plot_every: int = 100
    batch_size: int = 256
    replay_size: int = 200000
    warmup_steps: int = 5000
    updates_per_step: int = 1
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    hidden_dim: int = 256
    seed: int = 42
    device: str = 'auto'


class SACTrainer:
    def __init__(self, env, config, output_dir='sac_outputs'):
        self.env = env
        self.cfg = config
        self.output_dir = output_dir
        self.model_dir = os.path.join(output_dir, 'models')
        self.checkpoint_dir = os.path.join(self.model_dir, 'checkpoints')
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if config.device == 'auto' else config.device)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        self.actor = GaussianPolicy(env.state_dim, env.action_dim, env.max_acc, config.hidden_dim).to(self.device)
        self.q1 = QNetwork(env.state_dim, env.action_dim, config.hidden_dim).to(self.device)
        self.q2 = QNetwork(env.state_dim, env.action_dim, config.hidden_dim).to(self.device)
        self.q1_target = QNetwork(env.state_dim, env.action_dim, config.hidden_dim).to(self.device)
        self.q2_target = QNetwork(env.state_dim, env.action_dim, config.hidden_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=config.critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=config.critic_lr)
        self.log_alpha = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)
        self.target_entropy = -float(env.action_dim)

        self.replay = ReplayBuffer(config.replay_size, env.state_dim, env.action_dim, seed=config.seed + 99)
        self.total_steps = 0
        self.best_eval_score = -np.inf
        self.best_model_path = os.path.join(self.model_dir, 'best_sac_model.pt')
        self.latest_model_path = os.path.join(self.model_dir, 'latest_sac_model.pt')
        self.history_path = os.path.join(self.output_dir, 'sac_training_history.json')
        self.history_npz_path = os.path.join(self.output_dir, 'sac_training_history.npz')
        self.curves_path = os.path.join(self.output_dir, 'sac_training_curves.png')
        self.history = {
            'episodes': [], 'episode_rewards': [], 'episode_steps': [],
            'episode_path_lengths': [], 'episode_success': [], 'episode_collision': [],
            'eval_episodes': [], 'success_rates': [], 'collision_rates': [],
            'avg_steps': [], 'avg_path_lengths': [], 'avg_final_distances': [],
            'avg_min_obs_distances': [], 'eval_scores': [],
            'critic_losses': [], 'actor_losses': [], 'alpha_losses': [], 'alphas': [],
        }

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def soft_update(self, source, target):
        for src_p, tgt_p in zip(source.parameters(), target.parameters()):
            tgt_p.data.copy_(self.cfg.tau * src_p.data + (1.0 - self.cfg.tau) * tgt_p.data)

    def update(self):
        if self.replay.size < self.cfg.batch_size:
            return None
        s, a, r, s_next, done = self.replay.sample(self.cfg.batch_size, self.device)
        with torch.no_grad():
            next_a, next_logp, _ = self.actor.sample(s_next)
            q1_next = self.q1_target(s_next, next_a)
            q2_next = self.q2_target(s_next, next_a)
            q_next = torch.min(q1_next, q2_next) - self.alpha.detach() * next_logp
            target_q = r + (1.0 - done) * self.cfg.gamma * q_next

        q1_pred = self.q1(s, a)
        q2_pred = self.q2(s, a)
        q1_loss = F.mse_loss(q1_pred, target_q)
        q2_loss = F.mse_loss(q2_pred, target_q)
        self.q1_opt.zero_grad(); q1_loss.backward(); nn.utils.clip_grad_norm_(self.q1.parameters(), 5.0); self.q1_opt.step()
        self.q2_opt.zero_grad(); q2_loss.backward(); nn.utils.clip_grad_norm_(self.q2.parameters(), 5.0); self.q2_opt.step()

        new_a, logp, _ = self.actor.sample(s)
        q_pi = torch.min(self.q1(s, new_a), self.q2(s, new_a))
        actor_loss = (self.alpha.detach() * logp - q_pi).mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0); self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()

        self.soft_update(self.q1, self.q1_target)
        self.soft_update(self.q2, self.q2_target)

        return {
            'critic_loss': float((q1_loss + q2_loss).detach().cpu().item() * 0.5),
            'actor_loss': float(actor_loss.detach().cpu().item()),
            'alpha_loss': float(alpha_loss.detach().cpu().item()),
            'alpha': float(self.alpha.detach().cpu().item()),
        }

    def evaluate(self, max_steps=None):
        """在全新的随机静态/动态障碍物场景中评估当前策略。"""
        if max_steps is None:
            max_steps = self.cfg.steps_per_episode
        successes = collisions = 0
        total_steps = total_path = total_final = total_min = total_score = 0.0
        for _ in range(self.cfg.eval_scenarios):
            state = self.env.reset(randomize_static=True)
            min_obs = SENSOR_MAX_RANGE
            info = {'reached': False, 'collided': False, 'path_length': 0.0, 'dist_to_goal': np.inf}
            for step_i in range(max_steps):
                action = self.actor.act(state, self.device, deterministic=True)
                state, _, done, info = self.env.step(action)
                min_obs = min(min_obs, info['min_obs_dist'])
                if done:
                    break
            reached = info['reached']
            collided = info['collided']
            successes += int(reached)
            collisions += int(collided)
            total_steps += step_i + 1 if reached else max_steps
            total_path += info['path_length']
            total_final += info['dist_to_goal']
            total_min += min_obs
            total_score += (1000.0 * int(reached)
                            - 700.0 * int(collided)
                            - 45.0 * info['dist_to_goal']
                            - 20.0 * max(0.0, info['path_length'] - self.env.goal_distance)
                            - 0.5 * (step_i + 1))
        n = self.cfg.eval_scenarios
        return {
            'success_rate': successes / n,
            'collision_rate': collisions / n,
            'avg_steps': total_steps / n,
            'avg_path_length': total_path / n,
            'avg_final_dist': total_final / n,
            'avg_min_obs_dist': total_min / n,
            'avg_score': total_score / n,
        }

    def save_model(self, path, episode=None, tag=None):
        data = {
            'actor': self.actor.state_dict(),
            'q1': self.q1.state_dict(), 'q2': self.q2.state_dict(),
            'q1_target': self.q1_target.state_dict(), 'q2_target': self.q2_target.state_dict(),
            'actor_optimizer': self.actor_opt.state_dict(),
            'q1_optimizer': self.q1_opt.state_dict(),
            'q2_optimizer': self.q2_opt.state_dict(),
            'alpha_optimizer': self.alpha_opt.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu(),
            'episode': episode,
            'tag': tag,
            'total_steps': self.total_steps,
            'config': vars(self.cfg),
            'history': self.history,
            'best_eval_score': self.best_eval_score,
            'environment_config': {
                'start': self.env.start.tolist(), 'goal': self.env.goal.tolist(), 'dt': self.env.dt,
                'max_acc': self.env.max_acc, 'safe_margin': self.env.safe_margin,
                'max_speed': self.env.max_speed,
                'default_static_obstacles': [(o.center.tolist(), o.radius) for o in self.env.default_static_obstacles],
            }
        }
        torch.save(basic_metadata(data), path)

    def save_history(self):
        """实时保存训练数据，训练中断后也能直接画图或分析。"""
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.history_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
        np.savez(self.history_npz_path,
                 **{k: np.asarray(v) for k, v in self.history.items()})

    def save_checkpoint(self, episode, tag=None):
        """保存当前回合完整 SAC 模型；每次调用都会保留一个独立 checkpoint。"""
        if tag is None:
            tag = f'episode_{episode:05d}'
        checkpoint_path = os.path.join(self.checkpoint_dir, f'{tag}.pt')
        self.save_model(checkpoint_path, episode=episode, tag=tag)
        self.save_model(self.latest_model_path, episode=episode, tag='latest')
        return checkpoint_path

    def load_checkpoint(self, path, resume_training=True):
        """加载 SAC checkpoint；resume_training=True 时同时恢复优化器和历史曲线。"""
        data = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(data['actor'])
        if 'q1' in data:
            self.q1.load_state_dict(data['q1'])
        if 'q2' in data:
            self.q2.load_state_dict(data['q2'])
        if 'q1_target' in data:
            self.q1_target.load_state_dict(data['q1_target'])
        else:
            self.q1_target.load_state_dict(self.q1.state_dict())
        if 'q2_target' in data:
            self.q2_target.load_state_dict(data['q2_target'])
        else:
            self.q2_target.load_state_dict(self.q2.state_dict())
        if 'log_alpha' in data:
            loaded_alpha = data['log_alpha'].to(self.device)
            self.log_alpha.data.copy_(loaded_alpha)
        if resume_training:
            if 'actor_optimizer' in data:
                self.actor_opt.load_state_dict(data['actor_optimizer'])
            if 'q1_optimizer' in data:
                self.q1_opt.load_state_dict(data['q1_optimizer'])
            if 'q2_optimizer' in data:
                self.q2_opt.load_state_dict(data['q2_optimizer'])
            if 'alpha_optimizer' in data:
                self.alpha_opt.load_state_dict(data['alpha_optimizer'])
            if 'history' in data:
                self.history = data['history']
            self.total_steps = int(data.get('total_steps', self.total_steps))
            self.best_eval_score = float(data.get('best_eval_score', self.best_eval_score))
        print(f'已加载 SAC checkpoint: {path}', flush=True)
        print(f'恢复 total_steps={self.total_steps}, 历史回合数={len(self.history.get("episodes", []))}', flush=True)
        return data

    def load_actor(self, path):
        data = self.load_checkpoint(path, resume_training=False)
        self.actor.eval()
        return data

    def train(self):
        print('=' * 70)
        print('SAC 动态避障训练（随机静态障碍 + 随机动态障碍）')
        print('=' * 70)
        print(f'设备: {self.device}  状态维度: {self.env.state_dim}  动作维度: {self.env.action_dim}')
        print(f'输出目录: {self.output_dir}')
        print()

        start_episode = 0
        if self.history.get('episodes'):
            start_episode = int(self.history['episodes'][-1])
        end_episode = start_episode + self.cfg.episodes
        if start_episode > 0:
            print(f'续训模式: 从历史第 {start_episode} 回合后继续，计划训练到第 {end_episode} 回合', flush=True)

        recent_losses = deque(maxlen=200)
        for ep in range(start_episode + 1, end_episode + 1):
            state = self.env.reset(randomize_static=True)
            ep_reward = 0.0
            info = {'reached': False, 'collided': False, 'path_length': 0.0}
            for step_i in range(self.cfg.steps_per_episode):
                if self.total_steps < self.cfg.warmup_steps:
                    action = self.env.random_action()
                else:
                    action = self.actor.act(state, self.device, deterministic=False)
                next_state, reward, done, info = self.env.step(action)
                self.replay.add(state, action, reward, next_state, done)
                state = next_state
                ep_reward += reward
                self.total_steps += 1

                if self.total_steps >= self.cfg.warmup_steps:
                    for _ in range(self.cfg.updates_per_step):
                        loss_info = self.update()
                        if loss_info is not None:
                            recent_losses.append(loss_info)
                if done:
                    break

            self.history['episodes'].append(ep)
            self.history['episode_rewards'].append(ep_reward)
            self.history['episode_steps'].append(step_i + 1)
            self.history['episode_path_lengths'].append(info['path_length'])
            self.history['episode_success'].append(float(info['reached']))
            self.history['episode_collision'].append(float(info['collided']))

            if recent_losses:
                avg_loss = {k: float(np.mean([x[k] for x in list(recent_losses)])) for k in recent_losses[-1]}
                self.history['critic_losses'].append(avg_loss['critic_loss'])
                self.history['actor_losses'].append(avg_loss['actor_loss'])
                self.history['alpha_losses'].append(avg_loss['alpha_loss'])
                self.history['alphas'].append(avg_loss['alpha'])

            if ep % 20 == 0:
                print(f'[训练] 回合 {ep:5d}  奖励: {ep_reward:+8.2f}  步数: {step_i + 1:3d}  '
                      f'路径: {info["path_length"]:6.2f}  成功: {int(info["reached"])}  '
                      f'碰撞: {int(info["collided"])}  alpha: {float(self.alpha.detach().cpu()):.3f}  '
                      f'回放池: {self.replay.size}', flush=True)

            if ep % self.cfg.eval_every == 0:
                metrics = self.evaluate()
                self.history['eval_episodes'].append(ep)
                self.history['success_rates'].append(metrics['success_rate'])
                self.history['collision_rates'].append(metrics['collision_rate'])
                self.history['avg_steps'].append(metrics['avg_steps'])
                self.history['avg_path_lengths'].append(metrics['avg_path_length'])
                self.history['avg_final_distances'].append(metrics['avg_final_dist'])
                self.history['avg_min_obs_distances'].append(metrics['avg_min_obs_dist'])
                self.history['eval_scores'].append(metrics['avg_score'])
                print(f'[评估-随机] 回合 {ep:5d}  成功率: {metrics["success_rate"]:.1%}  '
                      f'碰撞率: {metrics["collision_rate"]:.1%}  平均步数: {metrics["avg_steps"]:.1f}  '
                      f'平均路径: {metrics["avg_path_length"]:.2f}  最终距离: {metrics["avg_final_dist"]:.3f}  '
                      f'得分: {metrics["avg_score"]:+.1f}', flush=True)
                if metrics['avg_score'] > self.best_eval_score:
                    self.best_eval_score = float(metrics['avg_score'])
                    self.save_model(self.best_model_path, episode=ep, tag='best')
                    print(f'  >>> 新 SAC 最佳模型已保存: {self.best_model_path} (得分 {self.best_eval_score:+.1f})', flush=True)

            if ep % self.cfg.save_every == 0:
                checkpoint_path = self.save_checkpoint(ep)
                print(f'  >>> SAC checkpoint 已保存: {checkpoint_path}', flush=True)

            self.save_history()
            if ep % self.cfg.plot_every == 0:
                plot_training_curves(self.history, self.curves_path)

        if not os.path.exists(self.best_model_path):
            self.save_model(self.best_model_path, episode=end_episode, tag='best_fallback')
        self.save_checkpoint(end_episode, tag='final')
        self.save_history()
        plot_training_curves(self.history, self.curves_path)
        print(f'\n训练完成。最佳 SAC 模型: {self.best_model_path}', flush=True)
        print(f'最新 SAC 模型: {self.latest_model_path}', flush=True)
        print(f'所有 checkpoint 目录: {self.checkpoint_dir}', flush=True)
        print(f'实时训练数据: {self.history_path}', flush=True)
        print(f'最佳评估得分: {self.best_eval_score:+.1f}', flush=True)


def moving_average(values, window=100):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    window = max(1, min(window, len(values)))
    return np.convolve(values, np.ones(window) / window, mode='valid')


def plot_training_curves(history, save_path):
    episodes = np.asarray(history.get('episodes', []), dtype=float)
    rewards = np.asarray(history.get('episode_rewards', []), dtype=float)
    eval_eps = np.asarray(history.get('eval_episodes', []), dtype=float)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    ax_reward, ax_rate, ax_path, ax_dist, ax_score, ax_loss = axes.ravel()

    if len(episodes):
        ax_reward.plot(episodes, rewards, color='#4C78A8', alpha=0.25, lw=0.8, label='Episode reward')
        ma = moving_average(rewards, 100)
        if len(ma):
            ax_reward.plot(episodes[len(episodes) - len(ma):], ma, color='#F58518', lw=2, label='100-episode MA')
    ax_reward.set_title('SAC Training Reward')
    ax_reward.set_xlabel('Episode'); ax_reward.set_ylabel('Reward'); ax_reward.grid(True, alpha=0.3); ax_reward.legend()

    if len(eval_eps):
        ax_rate.plot(eval_eps, np.asarray(history['success_rates']) * 100, 'o-', label='Success rate', color='#54A24B')
        ax_rate.plot(eval_eps, np.asarray(history['collision_rates']) * 100, 's--', label='Collision rate', color='#E45756')
    ax_rate.set_title('Random Evaluation Success / Collision')
    ax_rate.set_xlabel('Episode'); ax_rate.set_ylabel('Rate (%)'); ax_rate.set_ylim(-2, 102); ax_rate.grid(True, alpha=0.3); ax_rate.legend()

    if len(eval_eps):
        ax_path.plot(eval_eps, history['avg_steps'], 'o-', label='Avg steps', color='#B279A2')
        axp = ax_path.twinx()
        axp.plot(eval_eps, history['avg_path_lengths'], 's--', label='Avg path length', color='#FF9DA6')
        axp.set_ylabel('Path length')
        l1, lab1 = ax_path.get_legend_handles_labels(); l2, lab2 = axp.get_legend_handles_labels()
        ax_path.legend(l1 + l2, lab1 + lab2, loc='best')
    ax_path.set_title('Path Efficiency')
    ax_path.set_xlabel('Episode'); ax_path.set_ylabel('Steps'); ax_path.grid(True, alpha=0.3)

    if len(eval_eps):
        ax_dist.plot(eval_eps, history['avg_final_distances'], 'o-', label='Final distance', color='#F58518')
        ax_dist.plot(eval_eps, history['avg_min_obs_distances'], 's--', label='Min obstacle distance', color='#4C78A8')
    ax_dist.set_title('Goal Distance and Safety')
    ax_dist.set_xlabel('Episode'); ax_dist.set_ylabel('Distance'); ax_dist.grid(True, alpha=0.3); ax_dist.legend()

    if len(eval_eps):
        ax_score.plot(eval_eps, history['eval_scores'], 'o-', color='#72B7B2')
    ax_score.set_title('Validation Score')
    ax_score.set_xlabel('Episode'); ax_score.set_ylabel('Score'); ax_score.grid(True, alpha=0.3)

    loss_x = np.arange(1, len(history.get('critic_losses', [])) + 1)
    if len(loss_x):
        ax_loss.plot(loss_x, history['critic_losses'], label='Critic loss', color='#E45756', alpha=0.8)
        ax_loss.plot(loss_x, history['actor_losses'], label='Actor loss', color='#4C78A8', alpha=0.8)
        axa = ax_loss.twinx()
        axa.plot(loss_x, history['alphas'], label='Alpha', color='#54A24B', alpha=0.7)
        axa.set_ylabel('Alpha')
        l1, lab1 = ax_loss.get_legend_handles_labels(); l2, lab2 = axa.get_legend_handles_labels()
        ax_loss.legend(l1 + l2, lab1 + lab2, loc='best')
    ax_loss.set_title('SAC Loss / Temperature')
    ax_loss.set_xlabel('Logged update window'); ax_loss.set_ylabel('Loss'); ax_loss.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f'SAC 训练收敛图已保存: {save_path}', flush=True)


def rollout(env, actor, device, scenario=None, max_steps=600, deterministic=True):
    state = env.reset(randomize_static=(scenario is None), scenario=scenario)
    path = [env.agent_state.copy()]
    dob_trajs = [[] for _ in range(NUM_DYNAMIC_OBS)]
    for i, dob in enumerate(env.dynamic_obstacles):
        dob_trajs[i].append(dob.center.copy())
    info = {'reached': False, 'collided': False, 'dist_to_goal': np.inf, 'path_length': 0.0}
    for _ in range(max_steps):
        action = actor.act(state, device, deterministic=deterministic)
        state, _, done, info = env.step(action)
        path.append(env.agent_state.copy())
        for i, dob in enumerate(env.dynamic_obstacles):
            dob_trajs[i].append(dob.center.copy())
        if done:
            break
    return np.asarray(path), [np.asarray(t) for t in dob_trajs], info, clone_static_obstacles(env.static_obstacles)


def plot_trajectory(env, path, dob_trajs, static_obs, info, save_path):
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.scatter(*env.start[:2], c='green', s=140, marker='o', zorder=5, label='Start')
    ax.scatter(*env.goal, c='red', s=160, marker='*', zorder=5, label='Goal')
    theta = np.linspace(0, 2 * np.pi, 200)
    for i, obs in enumerate(static_obs):
        x = obs.center[0] + obs.radius * np.cos(theta)
        y = obs.center[1] + obs.radius * np.sin(theta)
        ax.fill(x, y, color='gray', alpha=0.7, zorder=2)
        ax.plot(x, y, 'k-', lw=2, label='Static obstacle' if i == 0 else None)
        sx = obs.center[0] + (obs.radius + env.safe_margin) * np.cos(theta)
        sy = obs.center[1] + (obs.radius + env.safe_margin) * np.sin(theta)
        ax.plot(sx, sy, 'k--', alpha=0.25, lw=1)
    colors = ['#FF6B6B', '#FFA500', '#FFD700']
    for i, traj in enumerate(dob_trajs):
        color = colors[i % len(colors)]
        ax.plot(traj[:, 0], traj[:, 1], '--', color=color, alpha=0.5, lw=1.2,
                label='Dynamic obstacle path' if i == 0 else None)
        ax.scatter(traj[0, 0], traj[0, 1], c=color, s=60, marker='s', edgecolors='black', zorder=3)
        ax.scatter(traj[-1, 0], traj[-1, 1], c='none', edgecolors=color, s=80, marker='o', zorder=3)
    ax.plot(path[:, 0], path[:, 1], 'b-', lw=2.5, label='SAC agent trajectory', zorder=4)
    ax.scatter(path[-1, 0], path[-1, 1], c='blue', s=110, marker='X', edgecolors='black', zorder=5)
    status = 'Reached' if info['reached'] else ('Collided' if info['collided'] else 'Timeout')
    txt = (f'Status: {status}\nSteps: {len(path)}  Path length: {info["path_length"]:.2f}\n'
           f'Distance to goal: {info["dist_to_goal"]:.3f}')
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va='top', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.set_title('SAC Dynamic Obstacle Avoidance')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.axis('equal'); ax.grid(True, alpha=0.3); ax.legend(loc='upper left', fontsize=9)
    all_x = [env.start[0], env.goal[0]] + list(path[:, 0])
    all_y = [env.start[1], env.goal[1]] + list(path[:, 1])
    for obs in static_obs:
        all_x.extend([obs.center[0] - obs.radius - 1, obs.center[0] + obs.radius + 1])
        all_y.extend([obs.center[1] - obs.radius - 1, obs.center[1] + obs.radius + 1])
    for traj in dob_trajs:
        all_x.extend(traj[:, 0]); all_y.extend(traj[:, 1])
    margin = 1.5
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f'SAC 轨迹图已保存: {save_path}')


def animate_trajectory(env, path, dob_trajs, static_obs, save_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(*env.start[:2], c='g', s=100, label='Start')
    ax.scatter(*env.goal, c='r', s=120, marker='*', label='Goal')
    theta = np.linspace(0, 2 * np.pi, 160)
    for i, obs in enumerate(static_obs):
        ax.fill(obs.center[0] + obs.radius * np.cos(theta),
                obs.center[1] + obs.radius * np.sin(theta),
                color='gray', alpha=0.6, label='Static obstacle' if i == 0 else None)
    colors = ['#FF6B6B', '#FFA500', '#FFD700']
    dob_circles, dob_lines = [], []
    for i in range(NUM_DYNAMIC_OBS):
        color = colors[i % len(colors)]
        circ = plt.Circle((0, 0), 0.4, color=color, alpha=0.55, zorder=3, label=f'Dynamic {i+1}' if i == 0 else None)
        ax.add_patch(circ); dob_circles.append(circ)
        line, = ax.plot([], [], '--', color=color, alpha=0.4, lw=1); dob_lines.append(line)
    agent_line, = ax.plot([], [], 'b-', lw=2.5, label='SAC agent')
    agent_point, = ax.plot([], [], 'bo', ms=9, zorder=5)
    all_x = [env.start[0], env.goal[0]] + list(path[:, 0])
    all_y = [env.start[1], env.goal[1]] + list(path[:, 1])
    for obs in static_obs:
        all_x.extend([obs.center[0] - obs.radius - 1, obs.center[0] + obs.radius + 1])
        all_y.extend([obs.center[1] - obs.radius - 1, obs.center[1] + obs.radius + 1])
    for traj in dob_trajs:
        all_x.extend(traj[:, 0]); all_y.extend(traj[:, 1])
    margin = 1.5
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin); ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.set_title('SAC Dynamic Obstacle Avoidance'); ax.set_xlabel('X'); ax.set_ylabel('Y')
    ax.axis('equal'); ax.grid(True, alpha=0.3); ax.legend(loc='upper left', fontsize=8)

    def update(frame):
        agent_line.set_data(path[:frame + 1, 0], path[:frame + 1, 1])
        agent_point.set_data([path[frame, 0]], [path[frame, 1]])
        for i in range(NUM_DYNAMIC_OBS):
            if frame < len(dob_trajs[i]):
                dob_circles[i].set_center(dob_trajs[i][frame])
                dob_lines[i].set_data(dob_trajs[i][:frame + 1, 0], dob_trajs[i][:frame + 1, 1])
        return [agent_line, agent_point] + dob_circles + dob_lines

    ani = FuncAnimation(fig, update, frames=len(path), interval=50, blit=True, repeat=False)
    ani.save(save_path, writer='pillow', fps=18)
    plt.close(fig)
    print(f'SAC 动图已保存: {save_path}')
