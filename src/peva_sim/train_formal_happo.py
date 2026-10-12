"""Formal recurrent HAPPO trainer with sequential factor correction."""
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

from .formal_happo import (
    FormalHAPPO, happo_surrogate_loss, update_advantage_factor)
from .formal_mappo import (
    ValueNorm, capture_environment, critic_state, features,
    restore_environment, sequence_batches, stack_rollout)
from .ocean_training_pool import OceanTrainingPool
from .ppo_math import disk_action, gae, latent_log_prob
from .protocol import file_sha
from .reproducibility import snapshot_source
from .separate_optimization import clipped_value_loss


ROOT = Path(__file__).resolve().parents[2]
VERSION = "formal_recurrent_happo_v1"


def atomic_torch_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def set_linear_lr(optimizer, base_lr, completed_updates, total_updates):
    factor = max(0.0, 1.0 - completed_updates / total_updates)
    for group in optimizer.param_groups:
        group["lr"] = base_lr * factor
    return factor


def tensor(value, device):
    return torch.as_tensor(
        np.asarray(value), dtype=torch.float32, device=device)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--rollout", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=64)
    parser.add_argument("--recurrent-chunk", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--gae-lambda", type=float, default=.95)
    parser.add_argument("--gamma", type=float, default=.99)
    parser.add_argument("--entropy-coef", type=float, default=.01)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--no-lr-decay", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pvf", action="store_true")
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--snapshot-every", type=int, default=50)
    parser.add_argument("--checkpoint-every-steps", type=int, default=0)
    parser.add_argument("--stop-after-steps", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"),
                        default="auto")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    positive = ("updates", "rollout", "epochs", "minibatch",
                "recurrent_chunk", "hidden_dim", "snapshot_every")
    if any(getattr(args, key) < 1 for key in positive):
        parser.error("training counts must be positive")
    if (args.rollout % args.recurrent_chunk or
            args.minibatch % args.recurrent_chunk):
        parser.error("rollout and minibatch must be divisible by recurrent chunk")
    if args.minibatch > args.rollout:
        parser.error("minibatch cannot exceed rollout")
    if not 0 <= args.gae_lambda <= 1 or not 0 < args.gamma <= 1:
        parser.error("invalid GAE parameters")
    for key in ("entropy_coef", "actor_lr", "critic_lr"):
        if not np.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            parser.error(key + " must be finite and nonnegative")
    if args.actor_lr == 0 or args.critic_lr == 0:
        parser.error("learning rates must be positive")
    if args.checkpoint_every_steps < 0 or args.stop_after_steps < 0:
        parser.error("checkpoint intervals must be nonnegative")
    return args


def checkpoint_contract(args):
    ignored = {"resume", "stop_after_steps", "checkpoint_every_steps"}
    return {key: value for key, value in vars(args).items()
            if key not in ignored}


def agent_logprob(model, observations, raw_actions, hidden_before, masks,
                  agent, chunk_length):
    """Re-evaluate one actor using the rollout's recurrent chunk boundary state."""
    length = observations.shape[0]
    if length % chunk_length:
        raise ValueError("rollout length is not divisible by recurrent chunk")
    result = torch.empty(length, dtype=observations.dtype,
                         device=observations.device)
    actor = model.actors.actors[agent]
    offsets = torch.arange(chunk_length, device=observations.device)
    for start in range(0, length, chunk_length):
        indices = start + offsets
        sequence_obs = observations[indices, agent][:, None, None, :]
        sequence_masks = masks[indices, agent][:, None, None, :]
        initial_hidden = hidden_before[start, agent][None, None, :]
        mean, logstd, _ = actor.sequence(
            sequence_obs, initial_hidden, sequence_masks, None)
        result[indices] = latent_log_prob(
            mean[:, 0, 0], logstd, raw_actions[indices, agent])
    return result


def main(argv=None):
    args = parse_args(argv)
    out = (ROOT / args.out).resolve()
    inventory_path = (ROOT / args.inventory).resolve()
    resume_path = (ROOT / args.resume).resolve() if args.resume else None
    if not out.is_relative_to(ROOT) or not inventory_path.is_relative_to(ROOT):
        raise SystemExit("paths must stay under project root")
    if resume_path is not None and not resume_path.is_relative_to(ROOT):
        raise SystemExit("resume checkpoint must stay under project root")
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

    def reset_episode():
        seed = int(episode_rng.integers(0, 2**31 - 1))
        observation, _ = env.reset(seed)
        return observation

    observation = reset_episode()
    obs_dim = features(observation, cfg, args.pvf).shape[-1]
    state_dim = len(critic_state(env))
    model = FormalHAPPO(
        cfg.n_uavs, obs_dim, state_dim, args.hidden_dim).to(device)
    value_norm = ValueNorm().to(device)
    actor_optimizers = [torch.optim.Adam(
        actor.parameters(), lr=args.actor_lr, eps=1e-5)
        for actor in model.actors.actors]
    critic_optimizer = torch.optim.Adam(
        model.critic.parameters(), lr=args.critic_lr, eps=1e-5)
    sampling_generator = torch.Generator(device=device).manual_seed(
        args.seed + 71003)
    order_generator = torch.Generator(device="cpu").manual_seed(
        args.seed + 72003)

    metadata = {
        "version": VERSION,
        "status": "formal candidate; publication status requires competence and protocol gates",
        "args": vars(args), "contract": checkpoint_contract(args),
        "config": env.config_dict(), "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "actor_parameters": [sum(parameter.numel() for parameter in actor.parameters())
                             for actor in model.actors.actors],
        "actor_inputs": (
            "per-agent local observation and fixed role; matched PVF or zero channel"),
        "actor_heterogeneity": "one independent recurrent actor per fixed UAV role",
        "critic_inputs": "privileged global state; training only",
        "sequential_update": (
            "random agent order per update; likelihood-ratio prefix factor recomputed "
            "after each agent's PPO epochs"),
        "communication": {"type": "none"},
        "mission_timeout": "finite-horizon terminal, zero bootstrap",
        "field_sha256": file_sha(Path(env.field.path)),
        "forcing": env.forcing_metadata(),
        "inventory_sha256": file_sha(inventory_path),
        "source_sha256": file_sha(__file__),
        "exact_resume_scope": (
            "model,optimizers,ValueNorm,all RNGs,environment,episode,RNN,unfinished rollout"),
    }

    update = 0
    total_steps = 0
    rollout_buffer = []
    completed = []
    episode_return = 0.0
    actor_hidden = model.actors.initial_state(device=device)
    actor_mask = torch.zeros(cfg.n_uavs, 1, device=device)
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
        model.load_state_dict(saved["model"])
        value_norm.load_state_dict(saved["value_norm"])
        for optimizer, state in zip(
                actor_optimizers, saved["actor_optimizers"], strict=True):
            optimizer.load_state_dict(state)
        critic_optimizer.load_state_dict(saved["critic_optimizer"])
        update = int(saved["update"])
        total_steps = int(saved["total_steps"])
        rollout_buffer = deepcopy(saved["rollout_buffer"])
        completed = deepcopy(saved["completed"])
        episode_return = float(saved["episode_return"])
        actor_hidden = saved["actor_hidden"].to(device)
        actor_mask = saved["actor_mask"].to(device)
        observation = restore_environment(env, saved["environment"])
        episode_rng.bit_generator.state = deepcopy(saved["episode_rng"])
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_cpu_rng"].cpu())
        if device.type == "cuda" and saved["torch_cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in saved["torch_cuda_rng"]])
        sampling_generator.set_state(saved["sampling_generator"].cpu())
        order_generator.set_state(saved["order_generator"].cpu())
        metadata = saved["metadata"]

    def exact_payload():
        return {
            "version": VERSION, "contract": checkpoint_contract(args),
            "metadata": metadata, "model": model.state_dict(),
            "value_norm": value_norm.state_dict(),
            "actor_optimizers": [optimizer.state_dict()
                                 for optimizer in actor_optimizers],
            "critic_optimizer": critic_optimizer.state_dict(),
            "update": update, "total_steps": total_steps,
            "rollout_buffer": deepcopy(rollout_buffer),
            "completed": deepcopy(completed),
            "episode_return": episode_return,
            "actor_hidden": actor_hidden.detach().cpu(),
            "actor_mask": actor_mask.detach().cpu(),
            "environment": capture_environment(env),
            "episode_rng": deepcopy(episode_rng.bit_generator.state),
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_cpu_rng": torch.get_rng_state(),
            "torch_cuda_rng": (torch.cuda.get_rng_state_all()
                               if device.type == "cuda" else None),
            "sampling_generator": sampling_generator.get_state().cpu(),
            "order_generator": order_generator.get_state().cpu(),
        }

    def save_exact(path=None):
        atomic_torch_save(exact_payload(), path or out / "resume.pt")

    while update < args.updates:
        rollout_reward = sum(float(row["r"]) for row in rollout_buffer)
        while len(rollout_buffer) < args.rollout:
            actor_features = features(observation, cfg, args.pvf)
            state = critic_state(env)
            with torch.no_grad():
                actor_input = tensor(actor_features, device)
                hidden_before = actor_hidden.clone()
                mask_before = actor_mask.clone()
                mean, logstd, actor_hidden = model.actors.step(
                    actor_input, actor_hidden, actor_mask)
                raw = mean + logstd.exp() * torch.randn(
                    mean.shape, dtype=mean.dtype, device=device,
                    generator=sampling_generator)
                logprob = latent_log_prob(mean, logstd, raw)
                action = disk_action(raw, cfg.speed_mps).cpu().numpy()
                value = value_norm.denormalize(
                    model.critic(tensor(state, device)).squeeze(-1))
                value_raw = float(value.cpu().item())
            scenario_id = env.active_scenario_id
            next_observation, reward, terminated, truncated, info = env.step(action)
            ended = bool(terminated or truncated)
            with torch.no_grad():
                if ended:
                    next_value_raw = 0.0
                else:
                    next_state = critic_state(env)
                    next_value = value_norm.denormalize(
                        model.critic(tensor(next_state, device)).squeeze(-1))
                    next_value_raw = float(next_value.cpu().item())
            rollout_buffer.append({
                "x": actor_features, "s": state, "raw": raw.cpu().numpy(),
                "lp": logprob.cpu().numpy(), "v": value_raw,
                "nv": next_value_raw, "r": reward,
                "boot": float(not ended), "cont": float(not ended),
                "hidden": hidden_before.cpu().numpy(),
                "mask": mask_before.cpu().numpy(),
            })
            rollout_reward += reward
            episode_return += reward
            total_steps += 1
            observation = next_observation
            actor_mask = torch.ones(cfg.n_uavs, 1, device=device)
            if ended:
                completed.append(dict(
                    info, return_value=episode_return,
                    environment_step=total_steps, scenario_id=scenario_id))
                episode_return = 0.0
                observation = reset_episode()
                actor_hidden.zero_()
                actor_mask.zero_()
            checkpoint_due = (args.checkpoint_every_steps and
                              total_steps % args.checkpoint_every_steps == 0)
            stopping = (args.stop_after_steps and
                        total_steps >= args.stop_after_steps)
            if checkpoint_due or stopping:
                save_exact()
            if stopping:
                (out / "episodes.json").write_text(json.dumps(completed, indent=2))
                env.close()
                return

        arrays = stack_rollout(rollout_buffer)
        advantage_np, returns_np = gae(
            arrays["r"], arrays["v"], arrays["nv"], arrays["boot"],
            arrays["cont"], gamma=args.gamma, lam=args.gae_lambda)
        advantage_np = ((advantage_np - advantage_np.mean()) /
                        (advantage_np.std() + 1e-8))
        value_norm.update(tensor(returns_np, device))

        observations = tensor(arrays["x"], device)
        states = tensor(arrays["s"], device)
        raw_actions = tensor(arrays["raw"], device)
        old_logprob = tensor(arrays["lp"], device)
        advantages = tensor(advantage_np, device)
        targets_norm = value_norm.normalize(tensor(returns_np, device))
        old_values_norm = value_norm.normalize(tensor(arrays["v"], device))
        hidden_before = tensor(arrays["hidden"], device)
        masks = tensor(arrays["mask"], device)

        with torch.no_grad():
            check = torch.stack([
                agent_logprob(model, observations, raw_actions, hidden_before,
                              masks, agent, args.recurrent_chunk)
                for agent in range(cfg.n_uavs)], dim=-1)
            ratio_error = float((torch.exp(check - old_logprob) - 1).abs().max())
        if ratio_error > 1e-5:
            raise RuntimeError("stored recurrent behavior probability inconsistent")

        lr_factor = 1.0
        if not args.no_lr_decay:
            for optimizer in actor_optimizers:
                lr_factor = set_linear_lr(
                    optimizer, args.actor_lr, update, args.updates)
            set_linear_lr(critic_optimizer, args.critic_lr, update, args.updates)

        critic_losses = []
        for _ in range(args.epochs):
            for starts in sequence_batches(
                    args.rollout, args.recurrent_chunk, args.minibatch,
                    sampling_generator, device):
                offsets = torch.arange(args.recurrent_chunk, device=device)
                indices = starts[:, None] + offsets[None, :]
                flat = indices.reshape(-1)
                predicted = model.critic(states[flat]).squeeze(-1)
                value_loss = clipped_value_loss(
                    predicted, old_values_norm[flat], targets_norm[flat])
                critic_optimizer.zero_grad(set_to_none=True)
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.critic.parameters(), 10.0)
                critic_optimizer.step()
                critic_losses.append(float(value_loss.detach()))

        order = torch.randperm(
            cfg.n_uavs, generator=order_generator).tolist()
        factor = torch.ones(args.rollout, dtype=torch.float32, device=device)
        actor_losses = []
        entropies = []
        for agent in order:
            actor = model.actors.actors[agent]
            optimizer = actor_optimizers[agent]
            for _ in range(args.epochs):
                for starts in sequence_batches(
                        args.rollout, args.recurrent_chunk, args.minibatch,
                        sampling_generator, device):
                    offsets = torch.arange(args.recurrent_chunk, device=device)
                    indices = starts[:, None] + offsets[None, :]
                    sequence_obs = observations[indices, agent].permute(
                        1, 0, 2)[:, :, None, :]
                    sequence_masks = masks[indices, agent].permute(
                        1, 0, 2)[:, :, None, :]
                    initial_hidden = hidden_before[starts, agent][:, None, :]
                    mean, logstd, _ = actor.sequence(
                        sequence_obs, initial_hidden, sequence_masks, None)
                    batch_raw = raw_actions[indices, agent].permute(1, 0, 2)
                    batch_old = old_logprob[indices, agent].permute(1, 0)
                    ratio = torch.exp(latent_log_prob(
                        mean[:, :, 0], logstd, batch_raw) - batch_old)
                    batch_factor = factor[indices].permute(1, 0)
                    batch_advantage = advantages[indices].permute(1, 0)
                    policy_loss = happo_surrogate_loss(
                        ratio, batch_factor, batch_advantage)
                    entropy = torch.distributions.Normal(
                        mean, logstd.exp()).entropy().sum(-1).mean()
                    objective = policy_loss - args.entropy_coef * entropy
                    optimizer.zero_grad(set_to_none=True)
                    objective.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
                    optimizer.step()
                    actor_losses.append(float(objective.detach()))
                    entropies.append(float(entropy.detach()))
            with torch.no_grad():
                new_logprob = agent_logprob(
                    model, observations, raw_actions, hidden_before, masks,
                    agent, args.recurrent_chunk)
                factor = update_advantage_factor(
                    factor, new_logprob, old_logprob[:, agent])

        update += 1
        row = {
            "update": update, "environment_steps": total_steps,
            "completed_episodes": len(completed),
            "rollout_reward": rollout_reward,
            "initial_ratio_max_error": ratio_error,
            "agent_order": order,
            "factor_mean": float(factor.mean()),
            "factor_min": float(factor.min()),
            "factor_max": float(factor.max()),
            "actor_loss_mean": float(np.mean(actor_losses)),
            "critic_loss_mean": float(np.mean(critic_losses)),
            "entropy_mean": float(np.mean(entropies)),
            "value_norm_mean": float(value_norm.mean),
            "value_norm_var": float(value_norm.var),
            "lr_factor": lr_factor,
            "elapsed_s": time.perf_counter() - started,
        }
        with (out / "updates.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        rollout_buffer.clear()
        save_exact()
        if update % args.snapshot_every == 0 or update == args.updates:
            atomic_torch_save({
                "version": VERSION, "model": model.state_dict(),
                "value_norm": value_norm.state_dict(), "metadata": metadata,
                "update": update, "environment_steps": total_steps,
            }, out / f"checkpoint-update{update:04d}.pt")

    atomic_torch_save({
        "version": VERSION, "model": model.state_dict(),
        "value_norm": value_norm.state_dict(), "metadata": metadata,
        "update": update, "environment_steps": total_steps,
    }, out / "checkpoint.pt")
    metadata["training_wall_s"] = time.perf_counter() - started
    metadata["environment_steps"] = total_steps
    metadata["updates_completed"] = update
    metadata["completed_episodes"] = len(completed)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (out / "episodes.json").write_text(json.dumps(completed, indent=2))
    env.close()


if __name__ == "__main__":
    main()
