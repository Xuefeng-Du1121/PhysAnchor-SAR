"""Formal MADDPG, MATD3, and MASAC trainer with exact resumability."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .formal_mappo import capture_environment, features, restore_environment
from .formal_offpolicy import (
    JointReplayBuffer, build_networks, radial_squash, update_offpolicy)
from .ocean_training_pool import OceanTrainingPool
from .protocol import file_sha
from .reproducibility import snapshot_source


ROOT = Path(__file__).resolve().parents[2]
VERSION = "formal_ctde_continuous_offpolicy_v1"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", required=True,
                        choices=("maddpg", "matd3", "masac"))
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=204800)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--replay-capacity", type=int, default=25000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=5000)
    parser.add_argument("--update-every", type=int, default=1)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--initial-alpha", type=float, default=.2)
    parser.add_argument("--gamma", type=float, default=.99)
    parser.add_argument("--tau", type=float, default=.005)
    parser.add_argument("--policy-delay", type=int, default=2)
    parser.add_argument("--policy-noise", type=float, default=.2)
    parser.add_argument("--noise-clip", type=float, default=.5)
    parser.add_argument("--exploration-noise", type=float, default=.1)
    parser.add_argument("--pvf", action="store_true")
    parser.add_argument("--pvf-anchor-strength", type=float, default=0.0,
                        help="physics-residual actor: latent PVF-peak anchor strength (0=off)")
    parser.add_argument("--pvf-anchor-angle-deg", type=float, default=15.0)
    parser.add_argument("--checkpoint-every-steps", type=int, default=25600)
    parser.add_argument("--stop-after-steps", type=int, default=0)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"),
                        default="auto")
    args = parser.parse_args(argv)
    positive = ("steps", "hidden_dim", "replay_capacity", "batch_size",
                "update_every", "actor_lr", "critic_lr", "alpha_lr",
                "initial_alpha", "gamma", "tau", "policy_delay")
    if any(getattr(args, name) <= 0 for name in positive):
        parser.error("training dimensions and coefficients must be positive")
    if args.batch_size > args.replay_capacity:
        parser.error("batch size exceeds replay capacity")
    if args.warmup_steps < args.batch_size:
        parser.error("warmup must contain at least one full minibatch")
    if args.warmup_steps >= args.steps:
        parser.error("warmup must be shorter than the training budget")
    if not 0 < args.gamma <= 1 or not 0 < args.tau <= 1:
        parser.error("invalid gamma or tau")
    for name in ("policy_noise", "noise_clip", "exploration_noise"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(name + " must be finite and nonnegative")
    if args.checkpoint_every_steps < 0 or args.stop_after_steps < 0:
        parser.error("checkpoint intervals must be nonnegative")
    return args


def checkpoint_contract(args):
    ignored = {"resume", "stop_after_steps", "checkpoint_every_steps"}
    return {key: value for key, value in vars(args).items()
            if key not in ignored}


def atomic_torch_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def network_state(networks):
    result = {
        "actor": networks.actor.state_dict(),
        "actor_target": networks.actor_target.state_dict(),
        "critic1": networks.critic1.state_dict(),
        "critic1_target": networks.critic1_target.state_dict(),
    }
    if networks.critic2 is not None:
        result.update({
            "critic2": networks.critic2.state_dict(),
            "critic2_target": networks.critic2_target.state_dict(),
        })
    return result


def load_network_state(networks, state):
    networks.actor.load_state_dict(state["actor"])
    networks.actor_target.load_state_dict(state["actor_target"])
    networks.critic1.load_state_dict(state["critic1"])
    networks.critic1_target.load_state_dict(state["critic1_target"])
    if networks.critic2 is not None:
        networks.critic2.load_state_dict(state["critic2"])
        networks.critic2_target.load_state_dict(state["critic2_target"])


def random_disk(rng, agents):
    angle = rng.uniform(0, 2 * np.pi, agents)
    radius = np.sqrt(rng.uniform(0, 1, agents))
    return np.column_stack((np.cos(angle), np.sin(angle))) * radius[:, None]


def main(argv=None):
    args = parse_args(argv)
    out = (ROOT / args.out).resolve()
    inventory_path = (ROOT / args.inventory).resolve()
    resume_path = (ROOT / args.resume).resolve() if args.resume else None
    if not out.is_relative_to(ROOT) or not inventory_path.is_relative_to(ROOT):
        raise SystemExit("paths must stay below the project root")
    if resume_path is not None and not resume_path.is_relative_to(ROOT):
        raise SystemExit("resume path must stay below the project root")
    if resume_path is None:
        if out.exists():
            raise SystemExit("refusing to overwrite output directory")
        out.mkdir(parents=True)
    elif not out.is_dir() or not resume_path.is_file():
        raise SystemExit("resume requires an existing output and checkpoint")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    env = OceanTrainingPool(inventory_path)
    env.value_density_enabled = args.pvf
    cfg = env.cfg
    episode_rng = np.random.default_rng(args.seed + 90000)
    exploration_rng = np.random.default_rng(args.seed + 91000)

    def reset_episode():
        seed = int(episode_rng.integers(0, 2**31 - 1))
        observation, _ = env.reset(seed)
        return observation

    observation = reset_episode()
    obs_dim = features(observation, cfg, args.pvf, args.pvf_anchor_strength > 0).shape[-1]
    if args.pvf_anchor_strength > 0 and not args.pvf:
        raise SystemExit("--pvf-anchor-strength requires --pvf")
    networks = build_networks(
        args.algorithm, cfg.n_uavs, obs_dim, args.hidden_dim,
        anchor_strength=args.pvf_anchor_strength,
        anchor_angle_deg=args.pvf_anchor_angle_deg,
        anchor_targets=cfg.n_targets)
    for value in vars(networks).values():
        if value is not None:
            value.to(device)
    actor_optimizer = torch.optim.Adam(
        networks.actor.parameters(), lr=args.actor_lr)
    critic_parameters = list(networks.critic1.parameters())
    if networks.critic2 is not None:
        critic_parameters += list(networks.critic2.parameters())
    critic_optimizer = torch.optim.Adam(
        critic_parameters, lr=args.critic_lr)
    log_alpha = (torch.tensor(np.log(args.initial_alpha), dtype=torch.float32,
                              device=device, requires_grad=True)
                 if args.algorithm == "masac" else None)
    alpha_optimizer = (torch.optim.Adam([log_alpha], lr=args.alpha_lr)
                       if log_alpha is not None else None)
    replay = JointReplayBuffer(
        args.replay_capacity, cfg.n_uavs, obs_dim, args.seed + 92000)

    metadata = {
        "version": VERSION,
        "algorithm": args.algorithm,
        "status": "formal candidate; claims require frozen multi-seed evaluation",
        "args": vars(args), "contract": checkpoint_contract(args),
        "config": env.config_dict(), "device": str(device),
        "actor_inputs": (
            "own position,time,local contacts,one-hot role; matched PVF or zero channel"),
        "critic_inputs": (
            "joint decentralized actor-feature rows and joint actions; training only"),
        "execution": "shared-parameter decentralized actor; no peer messages",
        "action_distribution": (
            "radially squashed Gaussian with exact Jacobian" if args.algorithm == "masac"
            else "deterministic radial disk policy with training exploration"),
        "mission_timeout": "finite-horizon terminal, zero bootstrap",
        "field_sha256": file_sha(Path(env.field.path)),
        "inventory_sha256": file_sha(inventory_path),
        "forcing": env.forcing_metadata(),
        "source_sha256": file_sha(__file__),
        "exact_resume_scope": (
            "networks,optimizers,replay,all RNGs,environment and unfinished episode"),
    }

    environment_steps = 0
    gradient_updates = 0
    episode_return = 0.0
    completed_episodes = []
    history = []
    started = time.perf_counter()

    if resume_path is None:
        metadata["source_archive_sha256"] = snapshot_source(out)
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    else:
        saved = torch.load(resume_path, map_location=device, weights_only=False)
        if saved.get("version") != VERSION:
            raise ValueError("unsupported exact-resume checkpoint")
        if saved["contract"] != checkpoint_contract(args):
            raise ValueError("resume arguments differ from checkpoint contract")
        if saved["metadata"]["inventory_sha256"] != file_sha(inventory_path):
            raise ValueError("inventory changed since checkpoint")
        load_network_state(networks, saved["networks"])
        actor_optimizer.load_state_dict(saved["actor_optimizer"])
        critic_optimizer.load_state_dict(saved["critic_optimizer"])
        if alpha_optimizer is not None:
            log_alpha.data.copy_(saved["log_alpha"].to(device))
            alpha_optimizer.load_state_dict(saved["alpha_optimizer"])
        replay.load_state_dict(saved["replay"])
        environment_steps = int(saved["environment_steps"])
        gradient_updates = int(saved["gradient_updates"])
        episode_return = float(saved["episode_return"])
        completed_episodes = deepcopy(saved["completed_episodes"])
        history = deepcopy(saved["history"])
        episode_rng.bit_generator.state = deepcopy(saved["episode_rng"])
        exploration_rng.bit_generator.state = deepcopy(saved["exploration_rng"])
        observation = restore_environment(env, saved["environment"])
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_cpu_rng"].cpu())
        if device.type == "cuda" and saved["torch_cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in saved["torch_cuda_rng"]])
        metadata = saved["metadata"]
        with (out / "resume-events.jsonl").open("a") as stream:
            stream.write(json.dumps({
                "checkpoint": str(resume_path.relative_to(ROOT)),
                "environment_steps": environment_steps,
                "gradient_updates": gradient_updates}) + "\n")

    def exact_payload():
        return {
            "version": VERSION, "contract": checkpoint_contract(args),
            "metadata": metadata, "networks": network_state(networks),
            "actor_optimizer": actor_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "log_alpha": (log_alpha.detach().cpu()
                          if log_alpha is not None else None),
            "alpha_optimizer": (alpha_optimizer.state_dict()
                                if alpha_optimizer is not None else None),
            "replay": replay.state_dict(),
            "environment_steps": environment_steps,
            "gradient_updates": gradient_updates,
            "episode_return": episode_return,
            "completed_episodes": deepcopy(completed_episodes),
            "history": deepcopy(history),
            "episode_rng": deepcopy(episode_rng.bit_generator.state),
            "exploration_rng": deepcopy(exploration_rng.bit_generator.state),
            "environment": capture_environment(env),
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_cpu_rng": torch.get_rng_state(),
            "torch_cuda_rng": (torch.cuda.get_rng_state_all()
                               if device.type == "cuda" else None),
        }

    def save_exact():
        atomic_torch_save(exact_payload(), out / "resume.pt")

    def inference_payload():
        return {
            "version": VERSION, "metadata": metadata,
            "actor": networks.actor.state_dict(),
            "environment_steps": environment_steps,
            "gradient_updates": gradient_updates,
            "log_alpha": (float(log_alpha.exp().detach())
                          if log_alpha is not None else None),
        }

    while environment_steps < args.steps:
        actor_features = features(observation, cfg, args.pvf, args.pvf_anchor_strength > 0)
        if environment_steps < args.warmup_steps and args.pvf_anchor_strength == 0:
            normalized_action = random_disk(exploration_rng, cfg.n_uavs)
        else:
            actor_tensor = torch.as_tensor(
                actor_features, dtype=torch.float32, device=device)
            with torch.no_grad():
                if args.algorithm == "masac":
                    normalized_action, _ = networks.actor.sample(actor_tensor)
                else:
                    normalized_action = networks.actor.deterministic(actor_tensor)
            normalized_action = normalized_action.cpu().numpy()
            if args.algorithm != "masac" and args.exploration_noise:
                normalized_action += exploration_rng.normal(
                    0, args.exploration_noise, normalized_action.shape)
                norm = np.linalg.norm(normalized_action, axis=-1, keepdims=True)
                normalized_action /= np.maximum(norm, 1.0 + 1e-6)

        next_observation, reward, terminated, truncated, info = env.step(
            normalized_action * cfg.speed_mps)
        done = bool(terminated or truncated)
        next_features = features(next_observation, cfg, args.pvf, args.pvf_anchor_strength > 0)
        replay.add(actor_features, normalized_action, reward, next_features,
                   0.0 if done else 1.0)
        environment_steps += 1
        episode_return += reward

        if (environment_steps >= args.warmup_steps and
                environment_steps % args.update_every == 0 and
                replay.size >= args.batch_size):
            gradient_updates += 1
            alpha = log_alpha.exp() if log_alpha is not None else .2
            update = update_offpolicy(
                args.algorithm, networks, replay.sample(args.batch_size, device),
                actor_optimizer, critic_optimizer, gradient_updates,
                gamma=args.gamma, tau=args.tau,
                policy_delay=args.policy_delay,
                policy_noise=args.policy_noise, noise_clip=args.noise_clip,
                alpha=alpha, log_alpha=log_alpha,
                alpha_optimizer=alpha_optimizer,
                target_entropy=-2.0 * cfg.n_uavs)
            if gradient_updates == 1 or gradient_updates % 100 == 0:
                row = {"environment_steps": environment_steps,
                       "gradient_updates": gradient_updates, **update}
                history.append(row)
                with (out / "updates.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)

        if done:
            completed_episodes.append({
                "environment_steps": environment_steps,
                "return_value": episode_return, "info": info,
                "scenario_id": env.active_scenario_id})
            observation = reset_episode()
            episode_return = 0.0
        else:
            observation = next_observation

        checkpoint_due = (args.checkpoint_every_steps and
                          environment_steps % args.checkpoint_every_steps == 0)
        stop_due = (args.stop_after_steps and
                    environment_steps >= args.stop_after_steps)
        if checkpoint_due or stop_due:
            save_exact()
        if stop_due and environment_steps < args.steps:
            env.close()
            print(json.dumps({"status": "stopped-for-resume",
                              "environment_steps": environment_steps}),
                  flush=True)
            return

    save_exact()
    atomic_torch_save(inference_payload(), out / "checkpoint.pt")
    metadata["training_wall_s"] = time.perf_counter() - started
    metadata["environment_steps"] = environment_steps
    metadata["gradient_updates"] = gradient_updates
    metadata["completed_episodes"] = len(completed_episodes)
    metadata["replay_size"] = replay.size
    metadata["final_alpha"] = (float(log_alpha.exp().detach())
                               if log_alpha is not None else None)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (out / "episodes.json").write_text(json.dumps(completed_episodes, indent=2))
    env.close()
    print(json.dumps({"status": "complete", "algorithm": args.algorithm,
                      "environment_steps": environment_steps,
                      "gradient_updates": gradient_updates}), flush=True)


if __name__ == "__main__":
    main()
