"""
Full 1400-episode fast-path SAC training run with NO early stopping.

This script trains from scratch using the fast-path SAC reward, even if the policy reaches
100% success before 1400 episodes. It also saves full SAC checkpoints, not only the actor:
  - actor
  - q1/q2 critics
  - target q1/q2 critics
  - entropy temperature log_alpha
  - all optimizer states

Output paths are supplied by the CLI. Random-number generator state is not
saved; these checkpoints are not exact-resumption snapshots.
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch

from . import core as sac
from . import fast


PROJECT_DIR = Path(__file__).resolve().parent
OUT_DIR = Path("runs/static2d")
CKPT_DIR = OUT_DIR / "checkpoints"


def full_checkpoint(agent, episode, total_steps, best_score, metrics=None):
    return {
        "episode": episode,
        "total_steps": total_steps,
        "best_score": best_score,
        "metrics": metrics or {},
        "actor": agent.actor.state_dict(),
        "q1": agent.q1.state_dict(),
        "q2": agent.q2.state_dict(),
        "q1_target": agent.q1_target.state_dict(),
        "q2_target": agent.q2_target.state_dict(),
        "log_alpha": agent.log_alpha.detach().cpu(),
        "actor_opt": agent.actor_opt.state_dict(),
        "q1_opt": agent.q1_opt.state_dict(),
        "q2_opt": agent.q2_opt.state_dict(),
        "alpha_opt": agent.alpha_opt.state_dict(),
        "gamma": agent.gamma,
        "tau": agent.tau,
    }


def save_full_checkpoint(agent, path, episode, total_steps, best_score, metrics=None):
    torch.save(full_checkpoint(agent, episode, total_steps, best_score, metrics), path)


def save_replay_buffer(replay, path):
    # Save only valid replay entries so continuation can be reproduced without wasting disk.
    np.savez_compressed(
        path,
        size=np.array([replay.size], dtype=np.int64),
        ptr=np.array([replay.ptr], dtype=np.int64),
        obs=replay.obs[:replay.size],
        act=replay.act[:replay.size],
        rew=replay.rew[:replay.size],
        next_obs=replay.next_obs[:replay.size],
        done=replay.done[:replay.size],
    )


def train(episodes=1400, max_steps=320, start_steps=3000, update_after=1000,
          update_every=1, eval_every=10, eval_episodes=20, eval_max_steps=420,
          seed=42, batch_size=256, replay_size=200000):
    """Train the original fast-path reward; timeouts are terminal as in the source."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = fast.FastPathAvoidanceEnv(seed=seed, max_steps=max_steps)
    agent = sac.SACAgent(env.obs_dim, env.act_dim, env.max_acc)
    replay = sac.ReplayBuffer(env.obs_dim, env.act_dim, capacity=replay_size)

    keys = ["episode", "total_steps", "train_reward", "train_success", "train_collision", "train_steps",
            "train_final_distance", "train_min_clearance", "train_path_length", "alpha",
            "eval_reward_mean", "eval_success_rate", "eval_collision_rate", "eval_steps_mean",
            "eval_final_distance_mean", "eval_min_clearance_mean", "eval_path_length_mean", "best_score"]

    csv_path = OUT_DIR / "training_log.csv"
    f = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    f.flush()

    logs = []
    total_steps = 0
    best_score = -np.inf
    best_episode = -1

    try:
        for ep in range(1, episodes + 1):
            obs = env.reset(randomize=True)
            ep_reward = 0.0
            min_clear = np.inf
            last_info = {"success": False, "collision": False, "distance": np.nan, "path_length": np.nan}
            last_update = {"alpha": float(agent.alpha.item())}

            for step in range(max_steps):
                if total_steps < start_steps:
                    action = env.rng.uniform(-env.max_acc, env.max_acc, size=env.act_dim).astype(np.float32)
                else:
                    action = agent.actor.act(obs, deterministic=False).astype(np.float32)

                next_obs, reward, done, info = env.step(action)
                replay.add(obs, action, reward, next_obs, float(done))
                obs = next_obs
                ep_reward += reward
                total_steps += 1
                min_clear = min(min_clear, info["clearance"])
                last_info = info

                if replay.size >= max(update_after, batch_size) and total_steps % update_every == 0:
                    last_update = agent.update(replay, batch_size=batch_size)

                if done:
                    break

            row = {
                "episode": ep,
                "total_steps": total_steps,
                "train_reward": float(ep_reward),
                "train_success": float(last_info["success"]),
                "train_collision": float(last_info["collision"]),
                "train_steps": step + 1,
                "train_final_distance": float(last_info["distance"]),
                "train_min_clearance": float(min_clear),
                "train_path_length": float(last_info["path_length"]),
                "alpha": float(last_update.get("alpha", agent.alpha.item())),
            }

            if ep % eval_every == 0 or ep == 1:
                metrics = fast.evaluate_fast(agent, episodes=eval_episodes, seed=7000 + ep, max_steps=eval_max_steps)
                row.update(metrics)
                score = (metrics["eval_success_rate"] * 3000.0
                         + metrics["eval_reward_mean"]
                         - 2.0 * metrics["eval_steps_mean"]
                         - 12.0 * metrics["eval_path_length_mean"]
                         - 120.0 * metrics["eval_collision_rate"]
                         - 5.0 * metrics["eval_final_distance_mean"])

                # Save a full checkpoint at every evaluation point.
                save_full_checkpoint(agent, CKPT_DIR / f"episode_{ep:04d}.pt",
                                     ep, total_steps, best_score, metrics)

                if score > best_score:
                    best_score = score
                    best_episode = ep
                    save_full_checkpoint(agent, OUT_DIR / "best_full.pt",
                                         ep, total_steps, best_score, metrics)
                    torch.save(agent.actor.state_dict(), OUT_DIR / "best_actor.pt")
                    with open(OUT_DIR / "best_info.json", "w", encoding="utf-8") as bf:
                        json.dump({"episode": ep, "score": best_score, "metrics": metrics}, bf)

                row["best_score"] = float(best_score)

            logs.append(row)
            writer.writerow({k: row.get(k, "") for k in keys})
            f.flush()

            if ep % 10 == 0 or ep == 1:
                msg = (f"ep={ep:4d} reward={ep_reward:8.2f} success={int(last_info['success'])} "
                       f"collision={int(last_info['collision'])} steps={step+1:3d} "
                       f"path={last_info['path_length']:.2f} alpha={row['alpha']:.3f}")
                if "eval_success_rate" in row:
                    msg += (f" eval_success={row['eval_success_rate']*100:5.1f}%"
                            f" eval_steps={row['eval_steps_mean']:6.1f}"
                            f" eval_path={row['eval_path_length_mean']:6.2f}"
                            f" eval_reward={row['eval_reward_mean']:8.2f}"
                            f" best_ep={best_episode}")
                print(msg, flush=True)
    finally:
        f.close()

    save_full_checkpoint(agent, OUT_DIR / "final_full.pt",
                         episodes, total_steps, best_score, logs[-1] if logs else {})
    torch.save(agent.actor.state_dict(), OUT_DIR / "final_actor.pt")
    save_replay_buffer(replay, OUT_DIR / "final_replay.npz")
    np.savez(OUT_DIR / "training_log.npz",
             **{k: np.array([r.get(k, np.nan) for r in logs], dtype=np.float32) for k in keys})
    return agent, logs
