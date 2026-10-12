"""Evaluate a checkpoint in a scenario with a different number of survivors.

v2 fixes v1, which passed the raw pre-squashing action mean to env.step as a
velocity in m/s. Training and the official evaluator map it through disk_action
onto the speed disk; v1 therefore flew UAVs at a fraction of v_max.
Also records the per-episode success rate directly (v1 summarised return only).
"""
import argparse, json
from pathlib import Path
import numpy as np, torch
from .mismatch_pool import MismatchTargetsPool
from .formal_mappo import FormalMAPPO, features, exchange_actor_messages
from .ppo_math import disk_action


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--inventory", required=True)
    p.add_argument("--n-targets", type=int, required=True, choices=[2, 5])
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=100)
    a = p.parse_args()

    out = Path(a.out)
    if (out / "summary.json").exists():
        raise SystemExit("refusing to overwrite " + str(out))
    out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    meta = ck["metadata"]
    env = MismatchTargetsPool(a.inventory, a.n_targets)
    cfg = env.cfg
    if meta["config"]["n_targets"] != cfg.n_targets:
        print(f"note: checkpoint trained with n_targets={meta['config']['n_targets']}, "
              f"evaluating with {cfg.n_targets}")

    use_pvf = bool(meta["args"].get("pvf", False))
    anchor = float(meta["args"].get("pvf_anchor_strength", 0.0))
    lc = "actor.message_head.weight" in ck["model"]
    obs_dim = (ck["model"]["actor.message_head.weight"].shape[1] if lc
               else ck["model"]["actor.encoder.0.weight"].shape[1])
    agent = FormalMAPPO(
        obs_dim, ck["model"]["critic.0.weight"].shape[1],
        ck["model"]["actor.encoder.0.weight"].shape[0],
        learned_comm=lc, role_dim=cfg.n_uavs,
        communication_type="commnet" if lc else "none",
        anchor_strength=anchor,
        anchor_angle_deg=meta["args"].get("pvf_anchor_angle_deg", 15.0),
        anchor_targets=cfg.n_targets)
    agent.load_state_dict(ck["model"])
    agent.eval()
    env.value_density_enabled = use_pvf

    rows = []
    for ep in range(a.episodes):
        obs, _ = env.reset(ep * 1000)
        hidden = agent.actor.initial_state(cfg.n_uavs)
        mask = torch.zeros(cfg.n_uavs, 1)
        done, ret, speeds = False, 0.0, []
        while not done:
            with torch.no_grad():
                x = torch.as_tensor(features(obs, cfg, use_pvf, anchor > 0), dtype=torch.float32)
                cm = exchange_actor_messages(agent.actor, x, env)[0] if lc else None
                mean, _, hidden = agent.actor.step(x, hidden, mask, cm)
                action = disk_action(mean, cfg.speed_mps).numpy()
            speeds.append(float(np.linalg.norm(action, axis=-1).mean()))
            obs, r, term, trunc, info = env.step(action)
            mask.fill_(1.0)
            ret += r
            done = term or trunc
        rows.append({"episode": ep, "return": ret, "mean_speed_mps": float(np.mean(speeds)),
                     "scenario_id": getattr(env, "active_scenario_id", None),
                     "info": {k: info[k] for k in ("success_rate", "ttd_all", "survival_score")}})
        print(json.dumps({"completed": ep + 1, "total": a.episodes}), flush=True)

    summary = {
        "version": "evaluate_mismatch_v2",
        "checkpoint": a.checkpoint, "n_targets_eval": cfg.n_targets,
        "n_targets_train": meta["config"]["n_targets"],
        "pvf": use_pvf, "anchor_strength": anchor, "episodes": a.episodes,
        "scenario_split": "train",
        "success_rate": float(np.mean([r["info"]["success_rate"] for r in rows])),
        "ttd_all": float(np.mean([r["info"]["ttd_all"] for r in rows])),
        "survival_score": float(np.mean([r["info"]["survival_score"] for r in rows])),
        "mean_speed_mps": float(np.mean([r["mean_speed_mps"] for r in rows])),
    }
    with open(out / "episodes.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
