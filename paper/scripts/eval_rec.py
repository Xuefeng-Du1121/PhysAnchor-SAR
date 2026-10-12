"""Held-out evaluation with a mis-specified number of anchor peaks, an
overridden anchor angle, or an explicitly parametrized classical planner.

* The environment contains G survivors (--true-targets); the public belief is
  initialized with the true G, exactly as in the benchmark.
* The anchor extracts K value peaks (--anchor-peaks); the default is the value
  the checkpoint was trained with.
* --zero-residual sets the learned action head and role biases to zero, which
  leaves the pure physics policy alpha R(theta_i) u_i; --angle overrides theta.
* --classical runs a non-learning planner; pvf-greedy and cbba accept an
  explicit travel scale and exclusion radius (the defaults reproduce
  peva_sim.classical_policies exactly).
Scenarios, seeds (820000 + 1000 * case + repeat) and environment construction
follow evaluate_formal_mappo / evaluate_formal_classical, so G = K = 3 with the
default settings reproduces the official summaries (checked by the guards).
"""
import _paths  # noqa: E402  (repo-relative experiment roots)
import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from peva_sim.ocean_training_pool import OceanTrainingPool
from peva_sim.ocean_forced_env import OceanForcedSAR
from peva_sim.evaluation_inventory import load_evaluation_cases
from peva_sim.formal_mappo import FormalMAPPO, features, exchange_actor_messages
from peva_sim.ppo_math import disk_action
from peva_sim.classical_policies import make_policy
from peva_sim.formal_mappo import physics_anchor_term
from peva_sim.protocol import file_sha
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extra_planners import EXTRA

ROOT = _paths.root("peva-phys-residual")
INV = ROOT / "configs/formal-ocean-forcing-inventory-v1.json"


def unit_velocity(delta, speed):
    delta = np.asarray(delta, dtype=np.float64)
    norm = np.linalg.norm(delta, axis=-1, keepdims=True)
    return (delta / np.maximum(norm, 1e-12) * speed).astype(np.float64)


class ParamGreedy:
    """classical_policies.PVFGreedy with explicit travel scale and exclusion radius."""

    def __init__(self, travel_m, exclusion_m):
        self.travel_m, self.exclusion_m = travel_m, exclusion_m

    def reset(self, observation):
        return None

    def act(self, observation, cfg, state=None):
        points = np.asarray(observation["prior_grid_m"], dtype=np.float64)
        positions = np.asarray(observation["positions_m"], dtype=np.float64)
        available = np.asarray(observation["value_density"], dtype=np.float64).copy()
        actions = []
        for position in positions:
            distance = np.linalg.norm(points - position, axis=1)
            score = available / (1 + distance / self.travel_m)
            selected = int(np.argmax(score))
            actions.append(points[selected] - position)
            radius2 = np.sum((points - points[selected]) ** 2, axis=1)
            available *= 1 - np.exp(-radius2 / (2 * self.exclusion_m ** 2))
        return unit_velocity(np.asarray(actions), cfg.speed_mps)


class ParamAuction:
    """classical_policies.CBBAPVF with explicit travel scale and exclusion radius."""

    def __init__(self, travel_m, exclusion_m):
        self.travel_m, self.exclusion_m = travel_m, exclusion_m

    def reset(self, observation):
        return None

    def act(self, observation, cfg, state=None):
        points = np.asarray(observation["prior_grid_m"], dtype=np.float64)
        positions = np.asarray(observation["positions_m"], dtype=np.float64)
        utility = np.asarray(observation["value_density"], dtype=np.float64)
        distance = np.linalg.norm(points[None, :, :] - positions[:, None, :], axis=-1)
        bids = utility[None, :] / (1.0 + distance / self.travel_m)
        available = np.ones(len(points), dtype=bool)
        targets = np.empty_like(positions)
        for agent in np.argsort(-np.max(bids, axis=1)):
            scores = bids[agent].copy()
            scores[~available] = -np.inf
            selected = int(np.argmax(scores))
            targets[agent] = points[selected]
            radius2 = np.sum((points - points[selected]) ** 2, axis=1)
            available &= radius2 > self.exclusion_m ** 2
            if not available.any():
                available[:] = True
        return unit_velocity(targets - positions, cfg.speed_mps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--classical")
    ap.add_argument("--physics-only", action="store_true",
                    help="pure anchor policy alpha R(theta_i) u_i without any network (any number of UAVs)")
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--n-uavs", type=int, default=6)
    ap.add_argument("--detection-p", type=float)
    ap.add_argument("--packet-drop", type=float)
    ap.add_argument("--drift-error-bound-mps", type=float, default=.1)
    ap.add_argument("--zero-residual", action="store_true")
    ap.add_argument("--angle", type=float)
    ap.add_argument("--true-targets", type=int, default=3)
    ap.add_argument("--anchor-peaks", type=int)
    ap.add_argument("--travel-scale-m", type=float, default=5000.0)
    ap.add_argument("--exclusion-mult", type=float, default=1.0)
    ap.add_argument("--split", default="test")
    ap.add_argument("--scenario-inventory")
    ap.add_argument("--episodes-per-scenario", type=int, default=50)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only-scenario", help="evaluate one scenario id only (case index and seeds unchanged)")
    ap.add_argument("--record-full", action="store_true",
                    help="with --record-episodes: also store value field, a particle subsample and the currents")
    ap.add_argument("--record-episodes", type=int, default=0,
                    help="also store full UAV/survivor trajectories of the first N repeats of every scenario")
    a = ap.parse_args()
    if sum(map(bool, (a.checkpoint, a.classical, a.physics_only))) != 1:
        ap.error("exactly one of --checkpoint / --classical / --physics-only is required")
    if a.checkpoint and a.n_uavs != 6:
        ap.error("learned checkpoints are trained for 6 UAVs; use --physics-only or --classical")
    out = Path(a.out)
    if (out / "summary.json").exists():
        raise SystemExit("refusing to overwrite " + str(out))
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    pool = OceanTrainingPool(INV)
    cfg3 = pool.cfg
    over = {k: v for k, v in (("detection_p", a.detection_p), ("packet_drop", a.packet_drop)) if v is not None}
    cfg3 = replace(cfg3, **over)
    cfg_env = replace(cfg3, n_targets=a.true_targets, n_uavs=a.n_uavs)
    scen = Path(a.scenario_inventory).resolve() if a.scenario_inventory else None
    cases, _ = load_evaluation_cases(INV, scen, a.split, pool.config_dict(), file_sha(Path(pool.field.path)))

    model, lc, anchor, angle, K, use_pvf, cfg_anchor = None, False, 0.0, None, None, True, cfg3
    ic3 = False
    if a.checkpoint:
        saved = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
        meta = saved["metadata"]
        trained_cfg = dict(meta["config"], n_targets=cfg3.n_targets)
        if trained_cfg != pool.config_dict() or meta["inventory_sha256"] != file_sha(INV):
            raise ValueError("checkpoint was not trained on the frozen benchmark")
        sd = saved["model"]
        ic3 = "actor.gate_head.weight" in sd
        lc = "actor.message_head.weight" in sd
    if a.checkpoint and ic3:
        from peva_sim.evaluate_formal_ic3net import load_model as load_ic3
        from peva_sim.formal_ic3net import exchange_gated_messages
        model, _ = load_ic3(saved, cfg3)
        use_pvf = bool(meta["args"]["pvf"])
        K = cfg3.n_targets
    elif a.checkpoint:
        anchor = float(meta["args"].get("pvf_anchor_strength", 0.0))
        angle = float(a.angle if a.angle is not None else meta["args"].get("pvf_anchor_angle_deg", 15.0))
        K = int(a.anchor_peaks or meta["config"]["n_targets"])
        use_pvf = bool(meta["args"]["pvf"])
        obs_dim = sd["actor.message_head.weight"].shape[1] if lc else sd["actor.encoder.0.weight"].shape[1]
        model = FormalMAPPO(
            obs_dim, sd["critic.0.weight"].shape[1], sd["actor.encoder.0.weight"].shape[0],
            learned_comm=lc, role_dim=cfg3.n_uavs,
            critic_mode="independent-local" if meta["args"].get("independent_critic") else "centralized",
            communication_type="commnet" if lc else "none",
            anchor_strength=anchor, anchor_angle_deg=angle, anchor_targets=K)
        model.load_state_dict(sd)
        if a.zero_residual:
            with torch.no_grad():
                model.actor.head.weight.zero_()
                model.actor.head.bias.zero_()
                if model.actor.role_bias is not None:
                    model.actor.role_bias.zero_()
        model.eval()
        cfg_anchor = replace(cfg3, n_targets=K)
    if a.physics_only:
        anchor, angle = a.alpha, float(30.0 if a.angle is None else a.angle)
        K = int(a.anchor_peaks or a.true_targets)
        cfg_anchor = replace(cfg_env, n_targets=K)
    exclusion = a.exclusion_mult * max(1.5 * cfg3.sensing_m, cfg3.region_m / cfg3.grid_side)

    def make_planner():
        if a.classical == "pvf-greedy":
            return ParamGreedy(a.travel_scale_m, exclusion)
        if a.classical == "cbba":
            return ParamAuction(a.travel_scale_m, exclusion)
        if a.classical in EXTRA:
            return EXTRA[a.classical]()
        return make_policy(a.classical)

    rows = []
    started = time.perf_counter()
    side = 21
    fax = np.linspace(0, cfg3.region_m, side)
    fgx, fgy = np.meshgrid(fax, fax)
    fpts = np.stack([fgx.ravel(), fgy.ravel()], -1)
    sub = np.random.default_rng(7).choice(cfg3.particles, 250, replace=False)
    for ci, case in enumerate(cases):
        if a.only_scenario and case["id"] != a.only_scenario:
            continue
        env = OceanForcedSAR(pool.field, case["origin_lonlat"], case["start_utc"], cfg_env,
                             truth_diffusion_range=(.5, 4.), drift_error_bound_mps=a.drift_error_bound_mps)
        if model is not None:
            env.value_density_enabled = use_pvf
        for rep in range(a.episodes_per_scenario):
            seed = 820000 + 1000 * ci + rep
            obs, _ = env.reset(seed)
            if model is not None:
                hidden = model.actor.initial_state(cfg3.n_uavs)
                mask = torch.zeros(cfg3.n_uavs, 1)
            elif not a.physics_only:
                planner = make_planner()
                planner.reset(obs)
            steps = 0
            rec = rep < a.record_episodes
            if rec:
                trU, trX, trF, trT = [], [], [], []
                full = {k: [] for k in ("pvf", "particles", "pweights", "flow", "action")}
            while not env.done:
                if rec:
                    trU.append(env.uavs.copy()); trX.append(env.targets.copy())
                    trF.append(env.found.copy()); trT.append(env.t)
                    if a.record_full:
                        full["pvf"].append(np.asarray(obs["value_density"], dtype=np.float64).copy())
                        full["particles"].append(env.particles[:, sub].copy())
                        full["pweights"].append(env.weights[:, sub].copy())
                        full["flow"].append(env.flow(fpts, env.t).reshape(side, side, 2))
                if a.physics_only:
                    x = torch.as_tensor(features(obs, cfg_anchor, True, True), dtype=torch.float32)
                    mean = physics_anchor_term(x, cfg_env.n_uavs, K, angle, anchor)
                    action = disk_action(mean, cfg_env.speed_mps).numpy()
                elif ic3:
                    with torch.no_grad():  # identical to evaluate_formal_ic3net
                        x = torch.as_tensor(features(obs, cfg3, use_pvf), dtype=torch.float32)
                        gate = (model.actor.gate_distribution(x, hidden, mask).probs >= .5).float()
                        delivery, _ = exchange_gated_messages(model.actor, x, gate, env)
                        mean, _, hidden = model.actor.step(x, hidden, mask, delivery)
                        action = disk_action(mean, cfg3.speed_mps).numpy()
                elif model is not None:
                    with torch.no_grad():
                        x = torch.as_tensor(features(obs, cfg_anchor, use_pvf, anchor > 0), dtype=torch.float32)
                        cm = exchange_actor_messages(model.actor, x, env)[0] if lc else None
                        mean, _, hidden = model.actor.step(x, hidden, mask, cm)
                        action = disk_action(mean, cfg3.speed_mps).numpy()
                else:
                    action = planner.act(obs, cfg_env, env.state() if a.classical == "oracle-target" else None)
                if rec and a.record_full:
                    full["action"].append(np.asarray(action, dtype=np.float64).copy())
                obs, _, _, _, info = env.step(action)
                if model is not None:
                    mask.fill_(1.0)
                steps += 1
            if rec:
                extra = {k: np.asarray(v) for k, v in full.items()} if a.record_full else {}
                if a.record_full:
                    extra.update(grid=np.asarray(obs["prior_grid_m"]), flow_axis=fax,
                                 origin=np.asarray(case["origin_lonlat"]), start_utc=case["start_utc"],
                                 sensing_m=cfg3.sensing_m, region_m=cfg3.region_m, radio_m=cfg3.radio_range_m,
                                 tau_s=cfg3.survival_tau_s, seed=seed)
                np.savez_compressed(out / f"traj-{case['id']}-rep{rep}.npz", uavs=np.asarray(trU),
                                    targets=np.asarray(trX), found=np.asarray(trF), t=np.asarray(trT),
                                    terminal_found=env.found.copy(), detected_at=env.detected_at.copy(), **extra)
            det = np.where(env.found, env.detected_at, np.inf)
            rows.append(dict(scenario_id=case["id"], case_index=ci, repeat=rep, seed=seed, steps=steps,
                             success_rate=float(info["success_rate"]), ttd_all=float(info["ttd_all"]),
                             survival_score=float(info["survival_score"]), all_found=float(env.found.all()),
                             detected_at=[float(v) if np.isfinite(v) else None for v in det]))
        print(json.dumps({"completed": len(rows), "total": len(cases) * a.episodes_per_scenario}), flush=True)

    def mean(key, sub=None):
        r = rows if sub is None else [x for x in rows if x["scenario_id"] == sub]
        return float(np.mean([x[key] for x in r]))

    summary = dict(
        version="eval_rec_v1", args=vars(a), split=a.split, true_targets=a.true_targets, anchor_peaks=K,
        n_uavs=a.n_uavs,
        anchor_angle_deg=angle, anchor_strength=anchor, zero_residual=a.zero_residual,
        checkpoint_sha256=file_sha(Path(a.checkpoint)) if a.checkpoint else None,
        planner_travel_scale_m=a.travel_scale_m if a.classical else None,
        planner_exclusion_m=exclusion if a.classical else None,
        episodes=len(rows), success_rate=mean("success_rate"), complete_rescue=mean("all_found"),
        ttd_all=mean("ttd_all"), survival_score=mean("survival_score"),
        per_scenario={c["id"]: dict(success_rate=mean("success_rate", c["id"]),
                                    complete_rescue=mean("all_found", c["id"])) for c in cases
                      if any(x["scenario_id"] == c["id"] for x in rows)},
        wall_s=time.perf_counter() - started)
    with (out / "episodes.jsonl").open("w") as stream:
        for r in rows:
            stream.write(json.dumps(r) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("success_rate", "complete_rescue", "ttd_all", "episodes")}))


if __name__ == "__main__":
    main()
