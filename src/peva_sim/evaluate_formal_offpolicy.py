"""Audited deterministic evaluator for formal off-policy CTDE baselines."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import platform
from pathlib import Path
import time

import numpy as np
import torch

from .audit_simulated_collection import verify_episode_arrays
from .belief_diagnostics import particle_diagnostics, summarize_particles
from .evaluation_inventory import EVALUATION_SPLITS, load_evaluation_cases
from .evaluation_trace import audit_trace
from .formal_mappo import features
from .formal_offpolicy import DecentralizedActor
from .ocean_forced_env import OceanForcedSAR
from .ocean_training_pool import OceanTrainingPool
from .protocol import file_sha


ROOT = Path(__file__).resolve().parents[2]
VERSION = "formal_ctde_continuous_offpolicy_v1"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--scenario-inventory", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", choices=EVALUATION_SPLITS,
                        default="validation")
    parser.add_argument("--episodes-per-scenario", type=int, default=10)
    parser.add_argument("--packet-drop", type=float, default=None)
    parser.add_argument("--radio-range-m", type=float, default=None)
    parser.add_argument("--detection-p", type=float, default=None)
    parser.add_argument("--clutter-mean", type=float, default=None)
    parser.add_argument("--belief-diffusion-m2s", type=float, default=None)
    parser.add_argument("--truth-diffusion-min-m2s", type=float, default=None)
    parser.add_argument("--truth-diffusion-max-m2s", type=float, default=None)
    parser.add_argument("--drift-error-bound-mps", type=float, default=None)
    args = parser.parse_args(argv)
    if args.episodes_per_scenario < 1:
        parser.error("episodes-per-scenario must be positive")
    for name in ("packet_drop", "detection_p"):
        value = getattr(args, name)
        if value is not None and not 0 <= value <= 1:
            parser.error(name.replace("_", "-") + " must be in [0,1]")
    if args.radio_range_m is not None and args.radio_range_m <= 0:
        parser.error("radio-range-m must be positive")
    for name in ("clutter_mean", "belief_diffusion_m2s",
                 "truth_diffusion_min_m2s", "truth_diffusion_max_m2s",
                 "drift_error_bound_mps"):
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(name.replace("_", "-") + " must be nonnegative")
    return args


def load_actor(saved, n_agents):
    if saved.get("version") != VERSION:
        raise ValueError("unsupported off-policy checkpoint architecture")
    metadata = saved.get("metadata", {})
    if metadata.get("version") != VERSION:
        raise ValueError("checkpoint metadata version mismatch")
    algorithm = metadata.get("algorithm")
    if algorithm not in ("maddpg", "matd3", "masac"):
        raise ValueError("unsupported off-policy algorithm")
    state = saved.get("actor")
    if not isinstance(state, dict) or "trunk.0.weight" not in state:
        raise ValueError("checkpoint does not contain an inference actor")
    hidden_dim, obs_dim = state["trunk.0.weight"].shape
    anchor_strength = float(metadata.get("args", {}).get("pvf_anchor_strength", 0.0))
    actor = DecentralizedActor(
        int(obs_dim), int(hidden_dim), stochastic=(algorithm == "masac"),
        anchor_strength=anchor_strength,
        anchor_angle_deg=metadata.get("args", {}).get("pvf_anchor_angle_deg", 15.0),
        anchor_targets=metadata.get("config", {}).get("n_targets", 3),
        role_dim=n_agents)
    actor.load_state_dict(state)
    actor.eval()
    if metadata.get("config", {}).get("n_uavs") != n_agents:
        raise ValueError("checkpoint team size differs from evaluation")
    return actor, metadata


def main(argv=None):
    args = parse_args(argv)
    truth_diffusion_range = (
        .5 if args.truth_diffusion_min_m2s is None
        else args.truth_diffusion_min_m2s,
        4. if args.truth_diffusion_max_m2s is None
        else args.truth_diffusion_max_m2s)
    if truth_diffusion_range[0] > truth_diffusion_range[1]:
        raise ValueError("truth diffusion minimum exceeds maximum")
    drift_error_bound_mps = (
        .1 if args.drift_error_bound_mps is None
        else args.drift_error_bound_mps)

    checkpoint, inventory_path, out = [
        (ROOT / value).resolve()
        for value in (args.checkpoint, args.inventory, args.out)]
    if not all(path.is_relative_to(ROOT)
               for path in (checkpoint, inventory_path, out)):
        raise ValueError("paths must stay under project root")
    scenario_inventory_path = (
        (ROOT / args.scenario_inventory).resolve()
        if args.scenario_inventory else None)
    if (scenario_inventory_path is not None and
            not scenario_inventory_path.is_relative_to(ROOT)):
        raise ValueError("scenario inventory must stay under project root")
    if out.exists():
        raise ValueError("refusing to overwrite evaluation output")

    torch.set_num_threads(1)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    pool = OceanTrainingPool(inventory_path)
    actor, metadata = load_actor(saved, pool.cfg.n_uavs)
    if metadata["config"] != pool.config_dict():
        raise ValueError("evaluation environment differs from training")
    if metadata["inventory_sha256"] != file_sha(inventory_path):
        raise ValueError("evaluation inventory differs from training")

    overrides = {key: value for key, value in (
        ("packet_drop", args.packet_drop),
        ("radio_range_m", args.radio_range_m),
        ("detection_p", args.detection_p),
        ("clutter_mean", args.clutter_mean),
        ("diffusion_m2s", args.belief_diffusion_m2s))
        if value is not None}
    cfg = replace(pool.cfg, **overrides)
    robustness_overrides = dict(overrides)
    if (args.truth_diffusion_min_m2s is not None or
            args.truth_diffusion_max_m2s is not None):
        robustness_overrides["truth_diffusion_range_m2s"] = list(
            truth_diffusion_range)
    if args.drift_error_bound_mps is not None:
        robustness_overrides["drift_error_bound_mps"] = drift_error_bound_mps
    cases, evaluation_inventory = load_evaluation_cases(
        inventory_path, scenario_inventory_path, args.split,
        pool.config_dict(), metadata["field_sha256"])

    out.mkdir(parents=True)
    source_manifest = {
        path.name: file_sha(path)
        for path in sorted((ROOT / "src/peva_sim").glob("*.py"))}
    (out / "source-manifest.json").write_text(
        json.dumps(source_manifest, indent=2), encoding="utf-8")
    rows = []
    evaluation_started = time.perf_counter()

    for case_index, case in enumerate(cases):
        env = OceanForcedSAR(
            pool.field, case["origin_lonlat"], case["start_utc"], cfg,
            truth_diffusion_range=truth_diffusion_range,
            drift_error_bound_mps=drift_error_bound_mps)
        env.value_density_enabled = metadata["args"]["pvf"]
        for repeat in range(args.episodes_per_scenario):
            seed = 820000 + 1000 * case_index + repeat
            observation, _ = env.reset(seed)
            times, found, actions, rewards = [], [], [], []
            positions, targets, inference_ms = [], [], []
            belief_means, belief_errors, belief_ess, belief_mass = [], [], [], []
            episode_started = time.perf_counter()
            total_reward = 0.0
            while not env.done:
                times.append(env.t)
                found.append(env.found.copy())
                state = env.state()
                positions.append(state["uavs_m"].copy())
                targets.append(state["targets_m"].copy())
                posterior_mean, error, ess, weight_mass, _ = particle_diagnostics(
                    env.particles, env.weights, state["targets_m"], env.found)
                belief_means.append(posterior_mean)
                belief_errors.append(error)
                belief_ess.append(ess)
                belief_mass.append(weight_mass)

                inference_started = time.perf_counter()
                with torch.no_grad():
                    actor_input = torch.as_tensor(
                        features(observation, cfg, metadata["args"]["pvf"],
                                 float(metadata.get("args", {}).get("pvf_anchor_strength", 0.0)) > 0),
                        dtype=torch.float32)
                    normalized_action = actor.deterministic(actor_input)
                    action = (normalized_action * cfg.speed_mps).numpy()
                inference_ms.append(
                    (time.perf_counter() - inference_started) * 1000)
                observation, reward, _, _, info = env.step(action)
                actions.append(action.copy())
                rewards.append(reward)
                total_reward += reward

            trace = out / ("episode-%04d.npz" % len(rows))
            np.savez_compressed(
                trace, time_s=np.asarray(times), found=np.asarray(found),
                terminal_found=env.found.copy(),
                actions_mps=np.asarray(actions), rewards=np.asarray(rewards),
                uavs_m=np.asarray(positions), targets_m=np.asarray(targets),
                belief_mean_m=np.asarray(belief_means),
                belief_error_m=np.asarray(belief_errors),
                belief_ess=np.asarray(belief_ess),
                belief_mass=np.asarray(belief_mass))
            record = {
                "scenario_id": case["id"], "seed": seed,
                "steps": len(times), "return_value": total_reward,
                "info": info, "trace_file": trace.name,
                "trace_sha256": file_sha(trace),
                "runtime": {
                    "episode_wall_s": time.perf_counter() - episode_started,
                    "inference_mean_ms": float(np.mean(inference_ms)),
                    "inference_p95_ms": float(np.percentile(inference_ms, 95)),
                    "inference_total_ms": float(np.sum(inference_ms)),
                },
            }
            verify_episode_arrays(
                np.asarray(times), np.asarray(found), record, vars(cfg),
                peer_payload_bytes=0)
            if not np.isclose(
                    total_reward, info["survival_score"] * cfg.n_targets,
                    rtol=1e-9, atol=1e-12):
                raise ValueError("reward/survival mismatch")
            with np.load(trace) as archive:
                audit_trace(archive, record, vars(cfg))
            trajectory = np.asarray(positions)
            margin = np.minimum(
                np.minimum(trajectory[..., 0], trajectory[..., 1]),
                np.minimum(cfg.region_m - trajectory[..., 0],
                           cfg.region_m - trajectory[..., 1]))
            record["trajectory_diagnostics"] = {
                "boundary_fraction_1km": float(np.mean(margin <= 1000.0)),
                "mean_speed_mps": float(np.linalg.norm(
                    np.asarray(actions), axis=-1).mean()),
                "commanded_mean_speed_mps": float(np.linalg.norm(
                    np.asarray(actions), axis=-1).mean()),
                "executed_mean_speed_mps": float(np.linalg.norm(
                    np.diff(trajectory, axis=0), axis=-1).mean() / cfg.dt_s)
                    if len(trajectory) > 1 else 0.0,
            }
            record["belief_diagnostics"] = summarize_particles(
                belief_errors, belief_ess, belief_mass, cfg.particles)
            rows.append(record)
            with (out / "episodes.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps({
                "completed": len(rows),
                "total": len(cases) * args.episodes_per_scenario,
            }), flush=True)
        env.close()

    summary = {
        "status": "formal protocol candidate; performance claim requires frozen multi-seed run",
        "split": args.split,
        "checkpoint_sha256": file_sha(checkpoint),
        "inventory_sha256": file_sha(inventory_path),
        **evaluation_inventory,
        "source_sha256": file_sha(__file__),
        "source_manifest_sha256": file_sha(out / "source-manifest.json"),
        "algorithm": metadata["algorithm"],
        "implementation_label": "parameter-shared " + metadata["algorithm"].upper(),
        "pvf": metadata["args"]["pvf"],
        "critic_mode": "centralized joint-action critic; training only",
        "training_steps": saved["environment_steps"],
        "gradient_updates": saved["gradient_updates"],
        "episodes": len(rows),
        "scenarios": [case["id"] for case in cases],
        "records": rows,
        "evaluation_config": vars(cfg),
        "environment_uncertainty": {
            "truth_diffusion_range_m2s": list(truth_diffusion_range),
            "drift_error_bound_mps": drift_error_bound_mps,
        },
        "robustness_overrides": robustness_overrides,
        "policy_communication": {
            "type": "none", "peer_payload_bytes_per_transmitting_agent_step": 0,
            "peer_header_bytes_per_transmitting_agent_step": 0,
            "shared_service": "coverage/PVF uplink and downlink counted by environment",
        },
        "evaluation_policy": (
            "deterministic radial actor output; MASAC uses the learned Gaussian mean"),
        "runtime": {
            "wall_s": time.perf_counter() - evaluation_started,
            "device": "cpu", "torch_threads": torch.get_num_threads(),
            "python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "platform": platform.platform(),
            "timing_scope": (
                "CPU feature construction and decentralized actor inference; "
                "excludes env.step and audit IO"),
        },
    }
    for key in ("success_rate", "ttd_all", "survival_score"):
        summary[key] = float(np.mean([row["info"][key] for row in rows]))
    summary["boundary_fraction_1km"] = float(np.mean([
        row["trajectory_diagnostics"]["boundary_fraction_1km"]
        for row in rows]))
    summary["geofence_interventions_mean"] = float(np.mean([
        row["info"]["geofence_interventions"] for row in rows]))
    summary["communication_bytes_mean"] = {
        key: float(np.mean([row["info"]["bytes"][key] for row in rows]))
        for key in rows[0]["info"]["bytes"]}
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    pool.close()


if __name__ == "__main__":
    main()
