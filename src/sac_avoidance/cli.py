"""Command-line entry points for training, evaluation and UAV simulation."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import platform

import numpy as np
import torch

from .policies import DIMENSIONS, export_policy, resolve_device

DEFAULTS = {
    "static2d": {"episodes": 1400, "max_steps": 320, "eval_every": 10, "eval_episodes": 20, "warmup_steps": 3000},
    "dynamic2d": {"episodes": 3000, "max_steps": 500, "eval_every": 100, "eval_episodes": 30, "warmup_steps": 5000},
    "dynamic3d": {"episodes": 5000, "max_steps": 600, "eval_every": 50, "eval_episodes": 10, "warmup_steps": 2000},
}


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def nonnegative(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return value


def build_parser():
    parser = argparse.ArgumentParser(description="SAC obstacle avoidance: 2D, 3D and quadrotor simulation.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["train", "evaluate", "uav"]:
        sub = commands.add_parser(name)
        sub.add_argument("--output", type=Path, help="New or empty output directory; defaults to a timestamped runs/ directory.")
        sub.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto" if name == "train" else "cpu")
        sub.add_argument("--threads", type=positive, default=1, help="PyTorch CPU threads (small networks often benefit from 1).")
        sub.add_argument("--seed", type=nonnegative, default=10000 if name == "evaluate" else 42)
        if name != "uav":
            sub.add_argument("--scenario", choices=list(DIMENSIONS), required=True)
            sub.add_argument("--episodes", type=positive, default=10 if name == "evaluate" else None)
            sub.add_argument("--max-steps", type=positive, default=600 if name == "evaluate" else None)
        if name == "train":
            sub.add_argument("--config", type=Path, help="JSON training overrides; explicit flags take precedence.")
            for option in ["eval-every", "eval-episodes", "batch-size", "replay-size"]:
                sub.add_argument(f"--{option}", type=positive)
            sub.add_argument("--warmup-steps", type=nonnegative)
        else:
            sub.add_argument("--checkpoint", type=Path, help="Portable policy.pt; defaults to the bundled pretrained actor.")
        if name == "uav":
            sub.add_argument("--duration", type=float, default=20.0, help="Simulation duration in seconds (at least 0.01).")
    return parser


def training_options(args):
    options = {**DEFAULTS[args.scenario], "batch_size": 256, "replay_size": 200000}
    if args.config:
        overrides = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(overrides, dict) or set(overrides) - set(options):
            raise ValueError(f"Config keys must be drawn from: {', '.join(options)}")
        options.update(overrides)
    for key in options:
        value = getattr(args, key)
        if value is not None:
            options[key] = value
        minimum = 0 if key == "warmup_steps" else 1
        if type(options[key]) is not int or options[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if options["replay_size"] < options["batch_size"]:
        raise ValueError("replay_size must be at least batch_size")
    return options


def train(args, options):
    """Wrap the established trainers and export the selected actor for inference.

    Full trainer checkpoints remain in the run directory. The portable policy
    does not imply exact resumption: replay buffers/RNG state are not restored.
    """
    output = args.output
    common = dict(episodes=options["episodes"], steps_per_episode=options["max_steps"],
                  eval_every=options["eval_every"], batch_size=options["batch_size"],
                  replay_size=options["replay_size"], seed=args.seed, device=args.device)
    if args.scenario == "static2d":
        from .static2d import core, training
        core.DEVICE = resolve_device(args.device)
        core.OUT_DIR = output
        training.OUT_DIR = output
        training.CKPT_DIR = output / "checkpoints"
        # The historical static reward includes its horizon and terminal timeout.
        agent, logs = training.train(
            episodes=options["episodes"], max_steps=options["max_steps"],
            start_steps=options["warmup_steps"], update_after=min(1000, options["warmup_steps"]),
            eval_every=options["eval_every"], eval_episodes=options["eval_episodes"],
            eval_max_steps=420, seed=args.seed, batch_size=options["batch_size"], replay_size=options["replay_size"],
        )
        core.plot_curves(logs)
        checkpoint = output / "best_full.pt"
        agent.actor.load_state_dict(torch.load(checkpoint, map_location=core.DEVICE, weights_only=True)["actor"])
        actor = agent.actor
    elif args.scenario == "dynamic2d":
        from .dynamic2d import SACAvoidanceEnv2D, SACConfig, SACTrainer
        cfg = SACConfig(**common, warmup_steps=options["warmup_steps"], eval_scenarios=options["eval_episodes"],
                        save_every=options["eval_every"], plot_every=options["eval_every"])
        trainer = SACTrainer(SACAvoidanceEnv2D(seed=args.seed), cfg, str(output))
        trainer.train()
        checkpoint = Path(trainer.best_model_path)
        trainer.load_actor(checkpoint)
        actor = trainer.actor
    else:
        from .dynamic3d import SACConfig, SAC3DAvoidanceTrainer
        cfg = SACConfig(**common, start_steps=options["warmup_steps"], eval_episodes=options["eval_episodes"], save_dir=str(output))
        trainer = SAC3DAvoidanceTrainer(cfg)
        trainer.train()
        checkpoint = Path(trainer.best_model_path)
        if not checkpoint.exists():
            checkpoint = output / "final_sac_model_3d.pt"
        trainer.load_model(checkpoint)
        actor = trainer.policy
    export_policy(actor, args.scenario, output / "policy.pt")
    return {"policy": "policy.pt", "selected_checkpoint": str(checkpoint.relative_to(output))}


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.device = str(resolve_device(args.device))
        torch.set_num_threads(args.threads)
        options = training_options(args) if args.command == "train" else None
        if args.command == "uav" and (not np.isfinite(args.duration) or args.duration < 0.01):
            raise ValueError("duration must be finite and at least 0.01 seconds")
        scenario = getattr(args, "scenario", "uav")
        if args.output is None:
            args.output = Path("runs") / scenario / f"{args.command}-{datetime.now():%Y%m%d-%H%M%S-%f}"
        args.output = args.output.resolve()
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError(f"Output directory is not empty: {args.output}. Choose a new --output path.")
        args.output.mkdir(parents=True, exist_ok=True)
        metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        metadata.update(resolved_training=options, python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__)
        (args.output / "run_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if args.command == "train":
            result = train(args, options)
        elif args.command == "evaluate":
            from .evaluation import evaluate
            result = evaluate(args.scenario, args.output, args.checkpoint, args.episodes, args.max_steps, args.seed, args.device)
        else:
            from .uav import simulate
            result = simulate(args.checkpoint, str(args.output), args.seed, args.duration, args.device)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"Output: {args.output}")
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
