"""Learning from a physics-policy initialization (behavior-cloning control).

pretrain: a recurrent MAPPO actor with the same inputs as the anchored actor
          (including the unit anchor direction u_i) but an anchor strength of
          1e-6, so its actions are produced by learned weights alone, is trained
          by DAgger to reproduce the pure physics policy alpha R(theta_i) u_i.
          Round 0 rolls out the physics policy; later rounds roll out the
          student and relabel with the analytic teacher. Training scenarios only.
finetune: the cloned actor initializes standard MAPPO training (unchanged
          peva_sim.train_formal_mappo); --warmup-updates freezes the actor for
          the first updates so the randomly initialized critic can fit first.
"""
import _paths  # noqa: E402  (repo-relative experiment roots)
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = _paths.root("peva-phys-residual")
INV = ROOT / "configs/formal-ocean-forcing-inventory-v1.json"


def pretrain(a):
    from peva_sim.ocean_training_pool import OceanTrainingPool
    from peva_sim.formal_mappo import (FormalMAPPO, features, exchange_actor_messages,
                                       physics_anchor_term)
    from peva_sim.ppo_math import disk_action
    from peva_sim.protocol import file_sha
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(1)
    skel = torch.load(a.skeleton, map_location="cpu", weights_only=False)
    meta, sd = skel["metadata"], skel["model"]
    if meta["args"]["pvf_anchor_strength"] > 1e-3:
        raise ValueError("skeleton must use a negligible anchor strength")
    env = OceanTrainingPool(INV)
    env.value_density_enabled = True
    cfg = env.cfg
    obs_dim = sd["actor.message_head.weight"].shape[1]
    model = FormalMAPPO(obs_dim, sd["critic.0.weight"].shape[1], sd["actor.encoder.0.weight"].shape[0],
                        learned_comm=True, role_dim=cfg.n_uavs, communication_type="commnet",
                        anchor_strength=meta["args"]["pvf_anchor_strength"],
                        anchor_angle_deg=meta["args"]["pvf_anchor_angle_deg"], anchor_targets=cfg.n_targets)
    model.load_state_dict(sd)
    actor = model.actor

    def teacher(x):
        return physics_anchor_term(x, cfg.n_uavs, cfg.n_targets, a.angle, a.strength)

    rng = np.random.default_rng(a.seed + 5150)

    def rollout(policy):
        seed = int(rng.integers(0, 2 ** 31 - 1))
        obs, _ = env.reset(seed)
        hidden = actor.initial_state(cfg.n_uavs)
        mask = torch.zeros(cfg.n_uavs, 1)
        X, C, M = [], [], []
        actor.eval()
        while not env.done:
            with torch.no_grad():
                x = torch.as_tensor(features(obs, cfg, True, True), dtype=torch.float32)
                cm = exchange_actor_messages(actor, x, env)[0]
                mu, _, hidden = actor.step(x, hidden, mask, cm)
                mean = teacher(x) if policy == "teacher" else mu
                action = disk_action(mean, cfg.speed_mps).numpy()
            X.append(x.numpy()); C.append(cm.numpy()); M.append(mask.numpy().copy())
            obs, _, _, _, info = env.step(action)
            mask = torch.ones(cfg.n_uavs, 1)
        return dict(x=np.asarray(X), c=np.asarray(C), m=np.asarray(M), found=float(info["success_rate"]))

    opt = torch.optim.Adam([p for n, p in actor.named_parameters() if n != "logstd"], lr=a.lr)

    def train(data, epochs):
        actor.train()
        for _ in range(epochs):
            order = rng.permutation(len(data))
            for b in range(0, len(order), a.batch):
                eps = [data[i] for i in order[b:b + a.batch]]
                L = max(len(e["x"]) for e in eps)
                B, A, D = len(eps), cfg.n_uavs, eps[0]["x"].shape[-1]
                X = np.zeros((L, B, A, D), np.float32)
                Cm = np.zeros((L, B, A, A), np.float32)
                Mk = np.zeros((L, B, A, 1), np.float32)
                V = np.zeros((L, B), np.float32)
                for j, e in enumerate(eps):
                    n = len(e["x"])
                    X[:n, j], Cm[:n, j], Mk[:n, j], V[:n, j] = e["x"], e["c"], e["m"], 1
                    # pad with the last valid step (keeps the one-hot role valid); masked out of the loss
                    X[n:, j], Cm[n:, j], Mk[n:, j] = e["x"][-1], e["c"][-1], 1
                Xt = torch.as_tensor(X)
                mu, _, _ = actor.sequence(Xt, torch.zeros(B, A, actor.hidden_dim), torch.as_tensor(Mk),
                                          torch.as_tensor(Cm))
                target = teacher(Xt)
                err = ((mu - target) ** 2).sum(-1).mean(-1)
                loss = (err * torch.as_tensor(V)).sum() / torch.as_tensor(V).sum()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                opt.step()
        return float(loss.item())

    def diagnose(data):
        actor.eval()
        with torch.no_grad():
            errs, ratios = [], []
            for e in data[-8:]:
                X = torch.as_tensor(e["x"])[:, None]
                mu, _, _ = actor.sequence(X, torch.zeros(1, cfg.n_uavs, actor.hidden_dim),
                                          torch.as_tensor(e["m"])[:, None], torch.as_tensor(e["c"])[:, None])
                t = teacher(X)
                cos = (mu * t).sum(-1) / (mu.norm(dim=-1) * t.norm(dim=-1)).clamp_min(1e-9)
                errs.append(torch.rad2deg(torch.acos(cos.clamp(-1, 1))).flatten())
                ratios.append((mu.norm(dim=-1) / t.norm(dim=-1).clamp_min(1e-9)).flatten())
        return float(torch.cat(errs).mean()), float(torch.cat(ratios).mean())

    data, log = [], []
    for rnd in range(a.rounds):
        policy = "teacher" if rnd == 0 else "student"
        new = [rollout(policy) for _ in range(a.episodes)]
        data += new
        loss = train(data, a.epochs)
        ang, ratio = diagnose(data)
        row = dict(round=rnd, policy=policy, episodes=len(data), rollout_found=float(np.mean([e["found"] for e in new])),
                   loss=loss, angle_error_deg=ang, norm_ratio=ratio)
        log.append(row)
        print(json.dumps(row), flush=True)
    final = [rollout("student") for _ in range(a.episodes)]
    row = dict(round="final-student", rollout_found=float(np.mean([e["found"] for e in final])))
    log.append(row)
    print(json.dumps(row), flush=True)
    skel["model"] = model.state_dict()
    skel["metadata"]["bc_initialization"] = dict(
        teacher=f"pure physics policy, alpha={a.strength}, theta={a.angle} deg (zero residual)",
        method="DAgger, sequence MSE on pre-squashing action mean", rounds=a.rounds, episodes_per_round=a.episodes,
        epochs=a.epochs, lr=a.lr, scenarios="training split only", skeleton=str(a.skeleton),
        skeleton_sha256=file_sha(Path(a.skeleton)), log=log,
        note="critic and ValueNorm are those of the 1-update skeleton; the evaluator uses the actor only")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(skel, out)
    print("saved", out)


def finetune(a, rest):
    import peva_sim.train_formal_mappo as T
    from peva_sim.formal_mappo import FormalMAPPO
    init = torch.load(a.init, map_location="cpu", weights_only=False)
    actor_state = {k[len("actor."):]: v for k, v in init["model"].items() if k.startswith("actor.")}

    class Initialized(FormalMAPPO):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.actor.load_state_dict(actor_state)

    T.FormalMAPPO = Initialized
    original = T.set_linear_lr
    calls = {"n": 0}

    def scheduled(optimizer, base_lr, completed, total):
        factor = original(optimizer, base_lr, completed, total)
        is_actor = calls["n"] % 2 == 0
        calls["n"] += 1
        if is_actor and completed < a.warmup_updates:
            for group in optimizer.param_groups:
                group["lr"] = 0.0
        return factor

    T.set_linear_lr = scheduled
    out = Path(rest[rest.index("--out") + 1])
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / (out.name + "-bc-finetune.json")).write_text(json.dumps(dict(
        init=str(a.init), warmup_updates=a.warmup_updates,
        bc_log=init["metadata"].get("bc_initialization")), indent=2))
    sys.argv = ["train_formal_mappo"] + rest
    T.main()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "pretrain":
        ap = argparse.ArgumentParser()
        ap.add_argument("--skeleton", required=True)
        ap.add_argument("--out", required=True)
        ap.add_argument("--seed", type=int, default=0)
        ap.add_argument("--angle", type=float, default=30.0)
        ap.add_argument("--strength", type=float, default=4.0)
        ap.add_argument("--rounds", type=int, default=6)
        ap.add_argument("--episodes", type=int, default=24)
        ap.add_argument("--epochs", type=int, default=40)
        ap.add_argument("--batch", type=int, default=8)
        ap.add_argument("--lr", type=float, default=1e-3)
        pretrain(ap.parse_args(sys.argv[2:]))
    elif mode == "finetune":
        # only --init / --warmup-updates are ours; everything else goes to train_formal_mappo
        ap = argparse.ArgumentParser()
        ap.add_argument("--init", required=True)
        ap.add_argument("--warmup-updates", type=int, default=0)
        known, rest = ap.parse_known_args(sys.argv[2:])
        finetune(known, rest)
    else:
        raise SystemExit("usage: bc_control.py pretrain|finetune ...")
