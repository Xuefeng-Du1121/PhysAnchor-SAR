"""Surface-current statistics inside every scenario box over the 3-h mission.

Speeds are sampled on a 21x21 lattice every 30 min; the 3-h drift is that of
nine passive drifters seeded on a 3x3 lattice and advected by the field alone.
"""
import _paths  # noqa: E402  (repo-relative experiment roots)
import json, sys
from pathlib import Path
import numpy as np
from peva_sim.ocean_field import OceanField
from peva_sim.real_env import LocalFrame

ROOT = _paths.root("peva-phys-residual")
inv = json.loads((ROOT / "configs/formal-ocean-forcing-inventory-v1.json").read_text())
f = OceanField(inv["field_path"])
R, H = inv["config"]["region_m"], inv["config"]["horizon_s"]
ax = np.linspace(0, R, 21)
gx, gy = np.meshgrid(ax, ax)
pts = np.stack([gx.ravel(), gy.ravel()], -1)
l3 = np.linspace(0.25, 0.75, 3) * R
seeds0 = np.stack(np.meshgrid(l3, l3), -1).reshape(-1, 2)
ids, splits, speeds, drift = [], [], [], []
for r in inv["scenarios"]:
    fr = LocalFrame(r["origin_lonlat"], R)
    ll = fr.inverse(pts)
    sp = [np.hypot(*f.query(r["start_utc"] + t, ll[:, 1], ll[:, 0]).T) for t in np.arange(0, H + 1, 1800.0)]
    x = seeds0.copy()
    for t in np.arange(0, H, 30.0):
        q = fr.inverse(x)
        x = x + f.query(r["start_utc"] + t, q[:, 1], q[:, 0]) * 30.0
    ids.append(r["id"]); splits.append(r["split"]); speeds.append(np.concatenate(sp))
    drift.append(np.linalg.norm(x - seeds0, axis=1))
    print(f"{r['id']:20s} {r['split']:10s} median speed {np.median(speeds[-1]):.3f} m/s, "
          f"p90 {np.percentile(speeds[-1], 90):.3f}, mean 3-h drift {drift[-1].mean() / 1000:.2f} km")
np.savez_compressed(Path(sys.argv[1]) / "scenario_speeds.npz", ids=np.array(ids), splits=np.array(splits),
                    speeds=np.stack(speeds), drift_m=np.stack(drift))
print("DONE")
