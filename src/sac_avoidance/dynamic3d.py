"""
PyTorch SAC 版本的 3D ADP 避障训练脚本。

复用 adpex_dynamic_3d.py 中的三维环境、传感器、奖励、评估逻辑思想，
但学习算法改为 Soft Actor-Critic (SAC)：
  - 高斯随机策略 + tanh squash，动作范围 [-max_acc, max_acc]
  - 双 Q critic + target critics
  - 自动温度 alpha 调整
  - 每回合保存 CSV，并绘制训练/诊断/评估曲线
"""

import csv
import os
from dataclasses import dataclass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .env3d import (
    AvoidanceEnv3D,
    NUM_DYNAMIC_OBS,
    SENSOR_MAX_RANGE,
)


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
EPS = 1e-6


class GaussianPolicy(nn.Module):
    """SAC 高斯策略网络：state -> mean/log_std -> tanh squash action。"""

    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super().__init__()
        self.max_action = float(max_action)
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        tanh_z = torch.tanh(z)
        action = tanh_z * self.max_action

        log_prob = normal.log_prob(z) - torch.log(
            self.max_action * (1.0 - tanh_z.pow(2)) + EPS
        )
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mean_action = torch.tanh(mean) * self.max_action
        return action, log_prob, mean_action


class QNetwork(nn.Module):
    """SAC Q 网络：concat(state, action) -> Q value。"""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class ReplayBuffer:
    """固定容量经验回放池。"""

    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = int(capacity)
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        self.state[self.pos] = state
        self.action[self.pos] = action
        self.reward[self.pos] = reward
        self.next_state[self.pos] = next_state
        self.done[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.state[idx], device=device),
            torch.as_tensor(self.action[idx], device=device),
            torch.as_tensor(self.reward[idx], device=device),
            torch.as_tensor(self.next_state[idx], device=device),
            torch.as_tensor(self.done[idx], device=device),
        )


@dataclass
class SACConfig:
    episodes: int = 5000
    steps_per_episode: int = 600
    eval_every: int = 50
    eval_episodes: int = 10
    save_dir: str = './models/sac'
    randomize_static: bool = True
    hidden_dim: int = 256
    replay_size: int = 200000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    alpha_lr: float = 3e-4
    train_every: int = 1
    updates_per_step: int = 1
    start_steps: int = 2000
    seed: int = 42
    device: str = 'auto'


class SAC3DAvoidanceTrainer:
    """SAC 训练器，复用 AvoidanceEnv3D 的环境和可视化。"""

    def __init__(self, config=None):
        self.cfg = config or SACConfig()
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        self.device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if self.cfg.device == 'auto' else self.cfg.device)

        self.env = AvoidanceEnv3D(
            start=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            goal=(7.0, 7.0, 7.0),
            dt=0.05,
            gamma=self.cfg.gamma,
            max_acc=1.2,
            safe_margin=0.65,
            seed=self.cfg.seed,
        )
        self.state_dim = self.env.state_dim
        self.action_dim = 3
        self.max_action = self.env.max_acc

        self.policy = GaussianPolicy(
            self.state_dim, self.action_dim, self.max_action, self.cfg.hidden_dim
        ).to(self.device)
        self.q1 = QNetwork(self.state_dim, self.action_dim, self.cfg.hidden_dim).to(self.device)
        self.q2 = QNetwork(self.state_dim, self.action_dim, self.cfg.hidden_dim).to(self.device)
        self.target_q1 = QNetwork(self.state_dim, self.action_dim, self.cfg.hidden_dim).to(self.device)
        self.target_q2 = QNetwork(self.state_dim, self.action_dim, self.cfg.hidden_dim).to(self.device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        self.policy_opt = optim.Adam(self.policy.parameters(), lr=self.cfg.policy_lr)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=self.cfg.q_lr)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=self.cfg.q_lr)
        self.log_alpha = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=self.cfg.alpha_lr)
        self.target_entropy = -float(self.action_dim)

        self.replay = ReplayBuffer(self.cfg.replay_size, self.state_dim, self.action_dim)
        self.total_steps = 0
        self.best_eval_score = -np.inf
        self.best_model_path = os.path.join(self.cfg.save_dir, 'best_sac_model_3d.pt')

        self.history = {
            'episode': [], 'episode_reward': [], 'episode_steps': [],
            'success': [], 'collision': [], 'min_obstacle_distance': [],
            'final_distance': [], 'q1_loss': [], 'q2_loss': [],
            'policy_loss': [], 'alpha_loss': [], 'alpha': [],
            'mean_log_prob': [], 'eval_episode': [], 'eval_success_rate': [],
            'eval_avg_steps': [], 'eval_avg_min_obstacle_distance': [],
            'eval_score': [],
        }

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state_features, deterministic=False):
        state = torch.as_tensor(state_features, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                _, _, action = self.policy.sample(state)
            else:
                action, _, _ = self.policy.sample(state)
        return action.cpu().numpy()[0]

    def _soft_update_targets(self):
        with torch.no_grad():
            for target, source in [(self.target_q1, self.q1), (self.target_q2, self.q2)]:
                for tp, sp in zip(target.parameters(), source.parameters()):
                    tp.data.mul_(1.0 - self.cfg.tau).add_(sp.data, alpha=self.cfg.tau)

    def update(self):
        state, action, reward, next_state, done = self.replay.sample(
            self.cfg.batch_size, self.device
        )

        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy.sample(next_state)
            target_q = torch.min(
                self.target_q1(next_state, next_action),
                self.target_q2(next_state, next_action),
            ) - self.alpha.detach() * next_log_prob
            target = reward + self.cfg.gamma * (1.0 - done) * target_q

        q1_pred = self.q1(state, action)
        q2_pred = self.q2(state, action)
        q1_loss = F.mse_loss(q1_pred, target)
        q2_loss = F.mse_loss(q2_pred, target)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        new_action, log_prob, _ = self.policy.sample(state)
        min_q = torch.min(self.q1(state, new_action), self.q2(state, new_action))
        policy_loss = (self.alpha.detach() * log_prob - min_q).mean()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self._soft_update_targets()
        return {
            'q1_loss': float(q1_loss.item()),
            'q2_loss': float(q2_loss.item()),
            'policy_loss': float(policy_loss.item()),
            'alpha_loss': float(alpha_loss.item()),
            'alpha': float(self.alpha.item()),
            'mean_log_prob': float(log_prob.mean().item()),
        }

    def _init_csv(self):
        os.makedirs(self.cfg.save_dir, exist_ok=True)
        with open(os.path.join(self.cfg.save_dir, 'training_history_3d.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode', 'episode_reward', 'episode_steps', 'success', 'collision',
                'min_obstacle_distance', 'final_distance', 'q1_loss', 'q2_loss',
                'policy_loss', 'alpha_loss', 'alpha', 'mean_log_prob', 'total_steps'
            ])
        with open(os.path.join(self.cfg.save_dir, 'evaluation_history_3d.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'success_rate', 'avg_steps', 'avg_min_obstacle_distance', 'eval_score'])

    def _append_training_csv(self):
        h = self.history
        i = -1
        with open(os.path.join(self.cfg.save_dir, 'training_history_3d.csv'), 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                h['episode'][i], h['episode_reward'][i], h['episode_steps'][i],
                h['success'][i], h['collision'][i], h['min_obstacle_distance'][i],
                h['final_distance'][i], h['q1_loss'][i], h['q2_loss'][i],
                h['policy_loss'][i], h['alpha_loss'][i], h['alpha'][i],
                h['mean_log_prob'][i], self.total_steps,
            ])

    def _append_eval_csv(self):
        h = self.history
        i = -1
        with open(os.path.join(self.cfg.save_dir, 'evaluation_history_3d.csv'), 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                h['eval_episode'][i], h['eval_success_rate'][i], h['eval_avg_steps'][i],
                h['eval_avg_min_obstacle_distance'][i], h['eval_score'][i],
            ])

    def train(self):
        self._init_csv()
        print(f'使用设备: {self.device}')
        print('SAC 探索方式: 策略随机采样 + 自动温度 alpha；没有手动衰减的外部高斯噪声。')

        for ep in range(self.cfg.episodes):
            self.env._reset_dynamic_obstacles()
            if self.cfg.randomize_static:
                self.env._randomize_static_obstacles()

            s_raw = self.env.start.copy()
            s_raw += np.array([0.05, 0.05, 0.05, 0.0, 0.0, 0.0]) * self.env.rng.normal(size=6)
            ep_reward = 0.0
            ep_steps = self.cfg.steps_per_episode
            ep_success = 0
            ep_collision = 0
            ep_min_dist = SENSOR_MAX_RANGE
            metric_sums = {k: 0.0 for k in ['q1_loss', 'q2_loss', 'policy_loss', 'alpha_loss', 'alpha', 'mean_log_prob']}
            update_count = 0

            for step_i in range(self.cfg.steps_per_episode):
                sf = self.env._state_features(s_raw).astype(np.float32)
                if self.total_steps < self.cfg.start_steps:
                    action = self.env.rng.uniform(-self.max_action, self.max_action, size=self.action_dim)
                else:
                    action = self.select_action(sf, deterministic=False)

                s_next_raw = self.env.step(s_raw, action)
                sf_next = self.env._state_features(s_next_raw).astype(np.float32)

                reached = (np.linalg.norm(s_next_raw[:3] - self.env.goal) < 0.40
                           and np.linalg.norm(s_next_raw[3:]) < 0.50)
                collided = self.env.obstacle_penalty(s_next_raw[:3]) >= 1000.0
                done = reached or collided

                for obs in self.env._all_obstacles():
                    d = np.linalg.norm(s_next_raw[:3] - obs.center) - obs.radius
                    ep_min_dist = min(ep_min_dist, d)

                if reached:
                    reward = 400.0
                elif collided:
                    reward = -200.0
                else:
                    reward = self.env.shaped_reward(s_raw, action, s_next_raw)
                ep_reward += reward

                self.replay.add(sf, action, reward, sf_next, done)
                s_raw = s_next_raw
                self.total_steps += 1

                if self.replay.size >= self.cfg.batch_size and step_i % self.cfg.train_every == 0:
                    for _ in range(self.cfg.updates_per_step):
                        metrics = self.update()
                        for k in metric_sums:
                            metric_sums[k] += metrics[k]
                        update_count += 1

                if done:
                    ep_steps = step_i + 1
                    ep_success = int(reached)
                    ep_collision = int(collided)
                    break

            avg_metrics = {k: metric_sums[k] / max(update_count, 1) for k in metric_sums}
            self.history['episode'].append(ep)
            self.history['episode_reward'].append(float(ep_reward))
            self.history['episode_steps'].append(int(ep_steps))
            self.history['success'].append(ep_success)
            self.history['collision'].append(ep_collision)
            self.history['min_obstacle_distance'].append(float(ep_min_dist))
            self.history['final_distance'].append(float(np.linalg.norm(s_raw[:3] - self.env.goal)))
            for k, v in avg_metrics.items():
                self.history[k].append(float(v))
            self._append_training_csv()

            if ep % 50 == 0:
                print(f'[SAC训练] 回合 {ep:4d} 奖励: {ep_reward:+.2f} '
                      f'成功:{ep_success} 碰撞:{ep_collision} alpha:{avg_metrics["alpha"]:.3f} '
                      f'回放池:{self.replay.size}')

            if ep % self.cfg.eval_every == 0 and ep > 0:
                succ, avg_steps, avg_min_d, score = self.evaluate(self.cfg.eval_episodes)
                self.history['eval_episode'].append(ep)
                self.history['eval_success_rate'].append(float(succ))
                self.history['eval_avg_steps'].append(float(avg_steps))
                self.history['eval_avg_min_obstacle_distance'].append(float(avg_min_d))
                self.history['eval_score'].append(float(score))
                self._append_eval_csv()
                print(f'[SAC评估] 回合 {ep:4d} 成功率:{succ:.1%} 平均步数:{avg_steps:.1f} '
                      f'最小障碍距离:{avg_min_d:.3f} 得分:{score:+.1f}')
                if score > self.best_eval_score:
                    self.best_eval_score = score
                    self.save_model(self.best_model_path)
                    print(f'  >>> 新最佳 SAC 模型已保存 (得分: {score:+.1f})')

        final_path = os.path.join(self.cfg.save_dir, 'final_sac_model_3d.pt')
        self.save_model(final_path)
        self.plot_training_curves()
        return self.history

    def evaluate(self, num_episodes=10, max_steps=None):
        max_steps = max_steps or self.cfg.steps_per_episode
        successes = 0
        total_steps = 0
        total_min_dist = 0.0
        total_score = 0.0

        for _ in range(num_episodes):
            self.env._reset_dynamic_obstacles()
            if self.cfg.randomize_static:
                self.env._randomize_static_obstacles()
            s = self.env.start.copy()
            min_dist = SENSOR_MAX_RANGE

            for step_count in range(max_steps):
                sf = self.env._state_features(s).astype(np.float32)
                action = self.select_action(sf, deterministic=True)
                s_next = self.env.step(s, action)

                for obs in self.env._all_obstacles():
                    d = np.linalg.norm(s_next[:3] - obs.center) - obs.radius
                    min_dist = min(min_dist, d)

                reached = (np.linalg.norm(s_next[:3] - self.env.goal) < 0.40
                           and np.linalg.norm(s_next[3:]) < 0.50)
                collided = self.env.obstacle_penalty(s_next[:3]) >= 1000.0
                if reached:
                    successes += 1
                    total_steps += step_count + 1
                    total_score += 1500.0 - step_count
                    break
                if collided:
                    total_score -= 500.0
                    break
                s = s_next
            else:
                final_dist = np.linalg.norm(s[:3] - self.env.goal)
                total_score += max(0, (1.0 - final_dist / self.env.state_scale)) * 500.0 - 200.0

            total_min_dist += min_dist

        n = num_episodes
        return successes / n, total_steps / max(successes, 1), total_min_dist / n, total_score / n

    def rollout(self, max_steps=800):
        self.env._reset_dynamic_obstacles()
        s = self.env.start.copy()
        path = [s.copy()]
        dob_trajs = [[] for _ in range(NUM_DYNAMIC_OBS)]
        for i, dob in enumerate(self.env.dynamic_obstacles):
            dob_trajs[i].append(dob.center.copy())

        for _ in range(max_steps):
            sf = self.env._state_features(s).astype(np.float32)
            action = self.select_action(sf, deterministic=True)
            s = self.env.step(s, action)
            path.append(s.copy())
            for i, dob in enumerate(self.env.dynamic_obstacles):
                dob_trajs[i].append(dob.center.copy())
            if (np.linalg.norm(s[:3] - self.env.goal) < 0.40
                    and np.linalg.norm(s[3:]) < 0.50):
                break
            if self.env.obstacle_penalty(s[:3]) >= 1000.0:
                break
        return np.array(path), [np.array(t) for t in dob_trajs]

    @staticmethod
    def moving_average(values, window):
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return values
        window = max(1, min(int(window), values.size))
        return np.convolve(values, np.ones(window) / window, mode='same')

    def plot_training_curves(self):
        h = self.history
        if not h['episode']:
            return []
        os.makedirs(self.cfg.save_dir, exist_ok=True)
        ep = np.asarray(h['episode'])
        window = min(40, max(3, len(ep) // 60))
        saved = []

        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        axes = axes.ravel()
        plots = [
            ('episode_reward', 'Reward Convergence', 'Reward', '#08519c'),
            ('episode_steps', 'Episode Steps', 'Steps', '#6a51a3'),
            ('final_distance', 'Final Distance to Goal', 'Distance', '#d94801'),
            ('min_obstacle_distance', 'Minimum Obstacle Clearance', 'Distance', '#006d77'),
        ]
        for ax, (key, title, ylabel, color) in zip(axes[[0, 2, 3, 4]], plots):
            y = np.asarray(h[key], dtype=float)
            ax.plot(ep, y, color=color, alpha=0.25, lw=0.8)
            ax.plot(ep, self.moving_average(y, window), color=color, lw=2, label=f'MA {window}')
            ax.set_title(title); ax.set_xlabel('Episode'); ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3); ax.legend()
        success = np.asarray(h['success'], dtype=float)
        collision = np.asarray(h['collision'], dtype=float)
        axes[1].plot(ep, self.moving_average(success, window) * 100, color='#238b45', lw=2, label='Success')
        axes[1].plot(ep, self.moving_average(collision, window) * 100, color='#cb181d', lw=2, label='Collision')
        axes[1].set_ylim(-2, 102); axes[1].set_title('Success / Collision Rate')
        axes[1].set_xlabel('Episode'); axes[1].set_ylabel('Rate (%)')
        axes[1].grid(True, alpha=0.3); axes[1].legend()
        alpha = np.asarray(h['alpha'], dtype=float)
        axes[5].plot(ep, alpha, color='#756bb1', lw=2, label='SAC alpha')
        axes[5].set_title('Entropy Temperature Alpha')
        axes[5].set_xlabel('Episode'); axes[5].set_ylabel('Alpha')
        axes[5].grid(True, alpha=0.3); axes[5].legend()
        fig.suptitle('3D SAC Training Curves', fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out = os.path.join(self.cfg.save_dir, 'sac_training_curves_3d.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved.append(out)

        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        axes = axes.ravel()
        diag = [
            ('q1_loss', 'Q1 Loss', '#e6550d'), ('q2_loss', 'Q2 Loss', '#fd8d3c'),
            ('policy_loss', 'Policy Loss', '#6a51a3'), ('alpha_loss', 'Alpha Loss', '#3182bd'),
            ('alpha', 'Alpha', '#756bb1'), ('mean_log_prob', 'Mean Log Prob', '#238b45'),
        ]
        for ax, (key, title, color) in zip(axes, diag):
            y = np.asarray(h[key], dtype=float)
            ax.plot(ep, y, color=color, alpha=0.25, lw=0.8)
            ax.plot(ep, self.moving_average(y, window), color=color, lw=2, label=f'MA {window}')
            ax.set_title(title); ax.set_xlabel('Episode')
            ax.grid(True, alpha=0.3); ax.legend()
        fig.suptitle('3D SAC Training Diagnostics', fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out = os.path.join(self.cfg.save_dir, 'sac_training_diagnostics_3d.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved.append(out)

        if h['eval_episode']:
            ee = np.asarray(h['eval_episode'])
            fig, axes = plt.subplots(2, 2, figsize=(13, 9))
            axes = axes.ravel()
            eval_specs = [
                ('eval_success_rate', 'Evaluation Success Rate', 'Success (%)', '#238b45', 100.0),
                ('eval_score', 'Evaluation Score', 'Score', '#08519c', 1.0),
                ('eval_avg_steps', 'Evaluation Average Steps', 'Steps', '#6a51a3', 1.0),
                ('eval_avg_min_obstacle_distance', 'Evaluation Minimum Clearance', 'Distance', '#006d77', 1.0),
            ]
            for ax, (key, title, ylabel, color, scale) in zip(axes, eval_specs):
                ax.plot(ee, np.asarray(h[key], dtype=float) * scale, 'o-', color=color, lw=2)
                ax.set_title(title); ax.set_xlabel('Episode'); ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.3)
            fig.tight_layout()
            out = os.path.join(self.cfg.save_dir, 'sac_training_evaluation_3d.png')
            fig.savefig(out, dpi=150, bbox_inches='tight')
            plt.close(fig)
            saved.append(out)
        print('SAC 曲线已保存:')
        for p in saved:
            print('  -', p)
        return saved

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy': self.policy.state_dict(),
            'q1': self.q1.state_dict(), 'q2': self.q2.state_dict(),
            'target_q1': self.target_q1.state_dict(), 'target_q2': self.target_q2.state_dict(),
            'policy_opt': self.policy_opt.state_dict(),
            'q1_opt': self.q1_opt.state_dict(), 'q2_opt': self.q2_opt.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu(),
            'alpha_opt': self.alpha_opt.state_dict(),
            'config': self.cfg.__dict__,
            'history': self.history,
            'total_steps': self.total_steps,
            'best_eval_score': self.best_eval_score,
        }, path)

    def load_model(self, path):
        data = torch.load(path, map_location=self.device, weights_only=True)
        self.policy.load_state_dict(data['policy'])
        self.q1.load_state_dict(data['q1']); self.q2.load_state_dict(data['q2'])
        self.target_q1.load_state_dict(data['target_q1']); self.target_q2.load_state_dict(data['target_q2'])
        self.log_alpha.data.copy_(data['log_alpha'].to(self.device))
        self.history = data.get('history', self.history)
        self.total_steps = data.get('total_steps', self.total_steps)
        self.best_eval_score = data.get('best_eval_score', self.best_eval_score)
