"""Numerical and integration checks; these do not assert that short training converges."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from sac_avoidance.env3d import AvoidanceEnv3D, generate_fibonacci_sphere_directions
from sac_avoidance.evaluation import run_episode
from sac_avoidance.policies import DIMENSIONS, export_policy, load_policy
from sac_avoidance.static2d.fast import FastPathAvoidanceEnv
from sac_avoidance.dynamic2d import SACAvoidanceEnv2D
from sac_avoidance.uav import QuadrotorModel, SACBridgeRateController


class NumericalChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_environment_shapes_and_bounds(self):
        """Observation dimensions and velocity limits must match saved networks."""
        static = FastPathAvoidanceEnv(seed=7, max_steps=1)
        self.assertEqual(static.reset().shape, (13,))
        obs, reward, done, info = static.step(np.array([100.0, -100.0]))
        self.assertTrue(done)
        self.assertTrue(info["timeout"])
        self.assertTrue(np.isfinite(obs).all())
        self.assertTrue(np.isfinite(reward))
        np.testing.assert_allclose(static.state[2:], [0.06, -0.06], atol=1e-7)

        dynamic = SACAvoidanceEnv2D(seed=7)
        self.assertEqual(dynamic.reset().shape, (18,))
        obs, reward, _, _ = dynamic.step(np.array([100.0, -100.0]))
        self.assertTrue(np.isfinite(obs).all())
        self.assertTrue(np.isfinite(reward))
        np.testing.assert_allclose(dynamic.agent_state[2:], [0.06, -0.06], atol=1e-7)

        spatial = AvoidanceEnv3D(seed=7)
        self.assertEqual(spatial._state_features(spatial.start).shape, (41,))
        np.testing.assert_allclose(np.linalg.norm(generate_fibonacci_sphere_directions(32), axis=1), 1.0, atol=1e-12)
        state = spatial.step(spatial.start, np.full(3, 100.0))
        np.testing.assert_allclose(state[3:], np.full(3, 0.06), atol=1e-7)
        for _ in range(100):
            state = spatial.step(state, np.full(3, 100.0))
        self.assertLessEqual(float(np.abs(state[3:]).max()), 2.0)

    def test_portable_policies(self):
        """All published actors load safely, remain bounded and round-trip exactly."""
        with tempfile.TemporaryDirectory() as tmp:
            for scenario, (obs_dim, act_dim) in DIMENSIONS.items():
                with self.subTest(scenario=scenario):
                    policy = load_policy(scenario)
                    obs = np.linspace(-0.5, 0.5, obs_dim, dtype=np.float32)
                    action = policy.act(obs)
                    self.assertEqual(action.shape, (act_dim,))
                    self.assertLessEqual(float(np.abs(action).max()), 1.2 + 1e-6)
                    export_policy(policy.actor, scenario, Path(tmp) / "policy.pt")
                    np.testing.assert_array_equal(action, load_policy(scenario, Path(tmp) / "policy.pt").act(obs))
                    state = torch.as_tensor(obs).unsqueeze(0).repeat(8, 1)
                    sampled, log_prob, _ = policy.actor.sample(state)
                    self.assertTrue(torch.isfinite(log_prob).all().item())
                    self.assertTrue(torch.isfinite(sampled).all().item())
                    with self.assertRaises(ValueError):
                        policy.act(np.zeros(obs_dim + 1))

    def test_manifest_and_wrong_scenario(self):
        path = load_policy("static2d").source.parent
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        for model in manifest.values():
            self.assertEqual(hashlib.sha256((path / model["file"]).read_bytes()).hexdigest(), model["sha256"])
        with self.assertRaises(ValueError):
            load_policy("dynamic3d", path / "static2d.pt")

    def test_seeded_rollouts(self):
        """Independent seeded runs reproduce states and don't confuse timeout with success."""
        for scenario in DIMENSIONS:
            policy = load_policy(scenario)
            metrics_a, geometry_a = run_episode(policy, 1234, 3)
            metrics_b, geometry_b = run_episode(policy, 1234, 3)
            self.assertEqual(metrics_a, metrics_b)
            np.testing.assert_array_equal(geometry_a[1], geometry_b[1])
            self.assertEqual(metrics_a["steps"], 3)
            self.assertEqual(metrics_a["outcome"], "timeout")

    def test_quadrotor_hover_and_coordinate_transform(self):
        """The motor mixer must balance gravity at a level hover."""
        model = QuadrotorModel()
        omega_hover = np.sqrt(model.m * model.g / (4.0 * model.Ct))
        state = np.zeros(17)
        state[6] = 1.0
        state[13:] = omega_hover
        throttle = np.full(4, (omega_hover - model.omega_b) / model.CR)
        state_dot = model.dynamics(state, lambda _: throttle)
        np.testing.assert_allclose(state_dot, np.zeros(17), atol=1e-10)
        vec = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(SACBridgeRateController.uav_to_adp_pos(SACBridgeRateController.adp_to_uav_pos(vec)), vec)


class CommandLineChecks(unittest.TestCase):
    def command(self, *args):
        process = subprocess.run([sys.executable, "-m", "sac_avoidance", *map(str, args)],
                                 capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180,
                                 env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        return process

    def test_train_evaluate_and_simulate(self):
        """Train with real gradient updates, export, reload and simulate each scenario."""
        with tempfile.TemporaryDirectory(prefix="sac smoke ") as tmp:
            for scenario in DIMENSIONS:
                train_dir = Path(tmp) / scenario
                self.command("train", "--scenario", scenario, "--episodes", 2, "--max-steps", 8,
                             "--eval-every", 1, "--eval-episodes", 1, "--batch-size", 4,
                             "--replay-size", 32, "--warmup-steps", 0, "--device", "cpu", "--output", train_dir)
                self.assertTrue((train_dir / "policy.pt").is_file())
                # Optimization must actually have run, not just collected transitions.
                if scenario == "static2d":
                    checkpoint = train_dir / "final_full.pt"
                    opt_key = "actor_opt"
                elif scenario == "dynamic2d":
                    checkpoint = train_dir / "models/latest_sac_model.pt"
                    opt_key = "actor_optimizer"
                else:
                    checkpoint = train_dir / "final_sac_model_3d.pt"
                    opt_key = "policy_opt"
                data = torch.load(checkpoint, map_location="cpu", weights_only=True)
                self.assertTrue(data[opt_key]["state"])
                for item in data[opt_key]["state"].values():
                    self.assertGreater(item["step"].item(), 0)
                    self.assertTrue(torch.isfinite(item["exp_avg"]).all().item())
                eval_dir = Path(tmp) / f"{scenario}-eval"
                self.command("evaluate", "--scenario", scenario, "--checkpoint", train_dir / "policy.pt",
                             "--episodes", 2, "--max-steps", 3, "--output", eval_dir)
                summary = json.loads((eval_dir / "summary.json").read_text())
                self.assertEqual(summary["episodes"], 2)
                self.assertTrue((eval_dir / "trajectory.png").is_file())
            uav_dir = Path(tmp) / "uav"
            self.command("uav", "--duration", 0.1, "--output", uav_dir)
            summary = json.loads((uav_dir / "summary.json").read_text())
            self.assertLess(summary["max_quaternion_norm_error"], 1e-10)
            self.assertGreaterEqual(summary["throttle_min"], 0.0)
            self.assertLessEqual(summary["throttle_max"], 1.0)

    def test_invalid_config_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "invalid"
            process = subprocess.run([sys.executable, "-m", "sac_avoidance", "train", "--scenario", "static2d",
                                      "--batch-size", "16", "--replay-size", "4", "--output", str(output)], capture_output=True)
            self.assertNotEqual(process.returncode, 0)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
