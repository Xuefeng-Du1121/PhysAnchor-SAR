"""Record visualization data for the paper (no metric is computed here).

1. Study-area currents: the Copernicus field (lon, lat, u, v) at the test start
   time and averaged over the month, plus the footprint of every scenario box.
2. One full-state replay of the first held-out test episode (test-jan22,
   seed 820000) with the theta=30 anchored policy (seed 3, final checkpoint):
   UAV/survivor positions, detections, public value field, NMS anchor peaks,
   a fixed particle subsample of the belief, radio links, and the local current
   field. The replay must reproduce the official evaluation trace exactly.
"""
import _paths  # noqa: E402  (repo-relative experiment roots)
import json, sys
from pathlib import Path
import numpy as np, torch

from peva_sim.ocean_training_pool import OceanTrainingPool
from peva_sim.ocean_forced_env import OceanForcedSAR
from peva_sim.real_env import LocalFrame
from peva_sim.formal_mappo import FormalMAPPO, features, exchange_actor_messages
from peva_sim.ppo_math import disk_action
from peva_sim.protocol import file_sha

ROOT = _paths.root("peva-phys-residual")
INV = ROOT / "configs/formal-ocean-forcing-inventory-v1.json"
CK = ROOT / "runs/campaign-v1/train/anchor-ang30-seed3/checkpoint-step-000204800.pt"
REF = ROOT / "runs/campaign-v1/eval/test/anchor-ang30-seed3/episode-0000.npz"
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)


def nms_peaks(points, values, cfg):
    """Same NMS as formal_mappo.pvf_anchor_directions, returning the peaks."""
    if not np.any(values > 0):
        return np.full((cfg.n_targets, 2), np.nan)
    available = values.astype(np.float64).copy()
    exclusion = max(1.5 * cfg.sensing_m, cfg.region_m / cfg.grid_side)
    peaks = []
    for _ in range(cfg.n_targets):
        k = int(np.argmax(available))
        peaks.append(points[k])
        available *= 1 - np.exp(-np.sum((points - points[k]) ** 2, axis=1) / (2 * exclusion ** 2))
    return np.asarray(peaks)


pool = OceanTrainingPool(INV)
cfg, field = pool.cfg, pool.field
inv = json.loads(INV.read_text())
saved = torch.load(CK, map_location="cpu", weights_only=False)
meta = saved["metadata"]
assert meta["config"] == pool.config_dict(), "config mismatch"
assert meta["inventory_sha256"] == file_sha(INV), "inventory mismatch"

# ---------------------------------------------------------------- study area
test_case = [r for r in inv["scenarios"] if r["id"] == "test-jan22"][0]
it0 = int(np.argmin(np.abs(field.time - test_case["start_utc"])))
boxes = {}
for r in inv["scenarios"]:
    fr = LocalFrame(r["origin_lonlat"], cfg.region_m)
    corners = np.array([[0, 0], [cfg.region_m, 0], [cfg.region_m, cfg.region_m], [0, cfg.region_m], [0, 0]])
    boxes[r["id"]] = fr.inverse(corners)
np.savez_compressed(
    OUT / "study_area.npz", lon=field.lon, lat=field.lat,
    u_test=field.u[it0], v_test=field.v[it0],
    u_mean=np.nanmean(field.u, axis=0), v_mean=np.nanmean(field.v, axis=0),
    box_ids=np.array(list(boxes)), boxes=np.stack(list(boxes.values())),
    test_start_utc=test_case["start_utc"])
print("study area saved; field time index", it0)

# ---------------------------------------------------------------- replay
lc = "actor.message_head.weight" in saved["model"]
anchor = float(meta["args"].get("pvf_anchor_strength", 0.0))
obs_dim = (saved["model"]["actor.message_head.weight"].shape[1] if lc
           else saved["model"]["actor.encoder.0.weight"].shape[1])
model = FormalMAPPO(
    obs_dim, saved["model"]["critic.0.weight"].shape[1],
    saved["model"]["actor.encoder.0.weight"].shape[0], learned_comm=lc,
    role_dim=cfg.n_uavs, critic_mode="centralized",
    communication_type="commnet" if lc else "none", anchor_strength=anchor,
    anchor_angle_deg=meta["args"].get("pvf_anchor_angle_deg", 15.0),
    anchor_targets=cfg.n_targets)
model.load_state_dict(saved["model"])
model.eval()
torch.set_num_threads(1)
print("anchor strength", anchor, "angle", meta["args"].get("pvf_anchor_angle_deg"), "comm", lc)

env = OceanForcedSAR(field, test_case["origin_lonlat"], test_case["start_utc"], cfg)
env.value_density_enabled = meta["args"]["pvf"]
obs, _ = env.reset(820000)
rng = np.random.default_rng(7)
sub = rng.choice(cfg.particles, 250, replace=False)
side = 21
axis = np.linspace(0, cfg.region_m, side)
gx, gy = np.meshgrid(axis, axis)
flow_pts = np.stack([gx.ravel(), gy.ravel()], -1)

hidden = model.actor.initial_state(cfg.n_uavs)
mask = torch.zeros(cfg.n_uavs, 1)
rec = {k: [] for k in ("t", "uavs", "targets", "found", "pvf", "peaks", "particles",
                       "pweights", "links", "flow", "anchor_dir", "action")}
while not env.done:
    st = env.state()
    rec["t"].append(env.t)
    rec["uavs"].append(st["uavs_m"].copy())
    rec["targets"].append(st["targets_m"].copy())
    rec["found"].append(env.found.copy())
    rec["pvf"].append(np.asarray(obs["value_density"], dtype=np.float64))
    rec["peaks"].append(nms_peaks(np.asarray(obs["prior_grid_m"]), np.asarray(obs["value_density"]), cfg))
    rec["particles"].append(env.particles[:, sub].copy())
    rec["pweights"].append(env.weights[:, sub].copy())
    rec["flow"].append(env.flow(flow_pts, env.t).reshape(side, side, 2))
    d = np.linalg.norm(st["uavs_m"][:, None] - st["uavs_m"][None], axis=-1)
    rec["links"].append(d <= cfg.radio_range_m)
    with torch.no_grad():
        x = torch.as_tensor(features(obs, cfg, meta["args"]["pvf"], anchor > 0), dtype=torch.float32)
        rec["anchor_dir"].append(x[:, -cfg.n_uavs - 2:-cfg.n_uavs].numpy().copy())
        cm = exchange_actor_messages(model.actor, x, env)[0] if lc else None
        mean, _, hidden = model.actor.step(x, hidden, mask, cm)
        action = disk_action(mean, cfg.speed_mps).numpy()
    rec["action"].append(action.copy())
    obs, _, _, _, info = env.step(action)
    mask.fill_(1.0)

arr = {k: np.asarray(v) for k, v in rec.items()}
ref = np.load(REF)
same = arr["uavs"].shape == ref["uavs_m"].shape and np.allclose(arr["uavs"], ref["uavs_m"])
print("replay steps", len(arr["t"]), "reference steps", len(ref["time_s"]), "identical trajectory:", same)
print("terminal found", env.found.tolist(), "detected_at_s", env.detected_at.tolist(), "info", info)
np.savez_compressed(OUT / "replay_anchor30_test_ep0.npz", **arr, grid=np.asarray(obs["prior_grid_m"]),
                    flow_axis=axis, terminal_found=env.found.copy(), detected_at=env.detected_at.copy(),
                    reproduces_official_trace=same, sensing_m=cfg.sensing_m, region_m=cfg.region_m,
                    radio_m=cfg.radio_range_m, origin=np.asarray(test_case["origin_lonlat"]))
if not same:
    raise SystemExit("replay does not reproduce the official trace")
print("DONE")
