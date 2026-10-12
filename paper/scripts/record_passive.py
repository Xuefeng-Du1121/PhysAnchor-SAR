"""Passive drift of the belief and of the survivors over the full 3-h horizon, with no search.

The UAVs are parked in a corner (zero velocity) far outside the prior, so no coverage evidence
or detection reshapes the belief: what remains is exactly the physics part of the problem --
advection by the Copernicus currents, diffusion, and survival decay of the value field.
Used for the introduction and method figures. Scenario: test-jan22 (seed 820000).
"""
import _paths  # noqa: E402  (repo-relative experiment roots)
import json
import sys
from dataclasses import replace
from pathlib import Path
import numpy as np
from peva_sim.ocean_training_pool import OceanTrainingPool
from peva_sim.ocean_forced_env import OceanForcedSAR
from peva_sim.formal_mappo import pvf_anchor_directions

ROOT = _paths.root("peva-phys-residual")
INV = ROOT / "configs/formal-ocean-forcing-inventory-v1.json"
OUT = Path(sys.argv[1])
pool = OceanTrainingPool(INV)
cfg = pool.cfg
inv = json.loads(INV.read_text())
SID = sys.argv[2] if len(sys.argv) > 2 else "test-jan22"
case = [r for r in inv["scenarios"] if r["id"] == SID][0]
env = OceanForcedSAR(pool.field, case["origin_lonlat"], case["start_utc"], cfg,
                     truth_diffusion_range=(.5, 4.), drift_error_bound_mps=.1)
obs, _ = env.reset(820000)
rng = np.random.default_rng(3)
sub = rng.choice(cfg.particles, 400, replace=False)
keep = {0, 60, 120, 240, 359}
snap = {k: [] for k in ("t", "pvf", "particles", "targets", "peaks_dir")}
tracks = []
park = np.full((cfg.n_uavs, 2), 300.0)
env.uavs[:] = park
side = 21
ax = np.linspace(0, cfg.region_m, side)
gx, gy = np.meshgrid(ax, ax)
fp = np.stack([gx.ravel(), gy.ravel()], -1)
flows = []
for k in range(int(round(cfg.horizon_s / cfg.dt_s))):
    env.uavs[:] = park
    tracks.append(env.targets.copy())
    if k in keep:
        snap["t"].append(env.t)
        snap["pvf"].append(np.asarray(env.observation()["value_density"]))
        snap["particles"].append(env.particles[:, sub].copy())
        snap["targets"].append(env.targets.copy())
        flows.append(env.flow(fp, env.t).reshape(side, side, 2))
    if env.done:
        break
    obs, _, _, _, info = env.step(np.zeros((cfg.n_uavs, 2)))
tr = np.asarray(tracks)
print("steps", len(tr), "found", env.found.tolist(), "target displacement km",
      np.round(np.linalg.norm(tr[-1] - tr[0], axis=1) / 1000, 2).tolist())
np.savez_compressed(OUT / f"passive_{SID}.npz", t=np.asarray(snap["t"]), pvf=np.asarray(snap["pvf"]),
                    particles=np.asarray(snap["particles"]), targets_snap=np.asarray(snap["targets"]),
                    tracks=tr, flow=np.asarray(flows), flow_axis=ax, grid=np.asarray(obs["prior_grid_m"]),
                    tau_s=cfg.survival_tau_s, detection_p=cfg.detection_p, region_m=cfg.region_m,
                    origin=np.asarray(case["origin_lonlat"]), start_utc=case["start_utc"])
print("DONE")
