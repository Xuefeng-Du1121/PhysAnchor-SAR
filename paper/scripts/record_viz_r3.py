"""Record what the new map figures need, from the official replay and the frozen inventory.

1. Ocean context for every scenario box: the Copernicus current field (u, v) on a fine lattice
   over each 40x40 km box at t0 and t0+3h, in the box's local metric frame.
2. For the first test episode replay (anchor theta=30, seed 3; already reproduces the official
   trace), the same local current field at every recorded step on a 21x21 lattice.
3. Per-split regional maps (lon/lat) for the train, validation, test and OOD scenario dates.
All values are read from the field; nothing is modelled here.
"""
import _paths  # noqa: E402  (repo-relative experiment roots)
import json
import sys
from pathlib import Path
import numpy as np
from peva_sim.ocean_field import OceanField
from peva_sim.real_env import LocalFrame

ROOT = _paths.root("peva-phys-residual")
INV = json.loads((ROOT / "configs/formal-ocean-forcing-inventory-v1.json").read_text())
OOD = json.loads((ROOT / "runs/complete-paper-v3/ood/ood-inventory.json").read_text())
OUT = Path(sys.argv[1])
f = OceanField(INV["field_path"])
R, H = INV["config"]["region_m"], INV["config"]["horizon_s"]

ax = np.linspace(0, R, 41)
gx, gy = np.meshgrid(ax, ax)
pts = np.stack([gx.ravel(), gy.ravel()], -1)


def local_uv(origin, t):
    fr = LocalFrame(origin, R)
    ll = fr.inverse(pts)
    uv = f.query(t, ll[:, 1], ll[:, 0])
    return uv.reshape(41, 41, 2)


rows = {r["id"]: r for r in INV["scenarios"]}
rows.update({r["id"]: r for r in OOD["scenarios"]})
box = {}
for sid, r in rows.items():
    box[sid + "|t0"] = local_uv(r["origin_lonlat"], r["start_utc"])
    box[sid + "|t3"] = local_uv(r["origin_lonlat"], r["start_utc"] + H)
np.savez_compressed(OUT / "box_currents.npz", axis_m=ax, ids=np.array(sorted(rows)),
                    origins=np.array([rows[k]["origin_lonlat"] for k in sorted(rows)]),
                    starts=np.array([rows[k]["start_utc"] for k in sorted(rows)]),
                    splits=np.array([rows[k]["split"] for k in sorted(rows)]),
                    **{k.replace("|", "__"): v for k, v in box.items()})

rp = np.load(OUT / "replay_anchor30_test_ep0.npz")
test = rows["test-jan22"]
ax21 = np.linspace(0, R, 21)
g2x, g2y = np.meshgrid(ax21, ax21)
p2 = np.stack([g2x.ravel(), g2y.ravel()], -1)
fr = LocalFrame(test["origin_lonlat"], R)
ll2 = fr.inverse(p2)
flow = np.stack([f.query(test["start_utc"] + t, ll2[:, 1], ll2[:, 0]).reshape(21, 21, 2) for t in rp["t"]])
np.savez_compressed(OUT / "replay_currents.npz", axis_m=ax21, flow=flow, t=rp["t"])

# regional maps at the start of each split's first date
lon, lat = np.asarray(f.lon), np.asarray(f.lat)
maps = {}
for tag, sid in (("train", "train-jan01-area0"), ("validation", "validation-jan12"), ("test", "test-jan22"),
                 ("ood", "ood-jan15-area0")):
    it = int(np.argmin(np.abs(f.time - rows[sid]["start_utc"])))   # raw grid: land stays NaN
    maps[tag + "_u"], maps[tag + "_v"] = f.u[it], f.v[it]
np.savez_compressed(OUT / "regional_maps.npz", lon=lon, lat=lat, **maps)
print("box scenarios", len(rows), "replay steps", len(rp["t"]), "DONE")
