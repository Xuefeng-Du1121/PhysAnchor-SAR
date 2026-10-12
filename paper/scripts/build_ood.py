"""Supplementary held-out scenarios: unused January 2018 dates x new sea areas.

The selection uses only the current field, never any policy result:
1. candidate box centres on a 0.75-degree lattice over the forcing domain;
2. keep boxes whose complete 3-h interpolation support is ocean on every date
   and whose geographic support is disjoint from every training, validation and
   test box (temporal disjointness follows from the dates);
3. rank by median current speed and take six areas at evenly spaced speed
   quantiles, at least 1.5 degrees apart;
4. dates 8, 15, 19 and 27 January 2018 at 00:00 UTC.
The inventory is then written by the unchanged peva_sim.build_forcing_inventory,
which re-checks land contact and cross-split disjointness.
"""
import _paths  # noqa: E402  (repo-relative experiment roots)
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from peva_sim.ocean_field import OceanField
from peva_sim.forcing_scenarios import support, intervals_overlap
from peva_sim.real_env import LocalFrame

ROOT = _paths.root("peva-phys-residual")
INV = ROOT / "configs/formal-ocean-forcing-inventory-v1.json"
OUT = Path(sys.argv[1]).resolve()
OUT.mkdir(parents=True, exist_ok=True)
inv = json.loads(INV.read_text())
field = OceanField(inv["field_path"])
R, H = inv["config"]["region_m"], inv["config"]["horizon_s"]
JAN1 = 1514764800
DATES = [JAN1 + (d - 1) * 86400 for d in (8, 15, 19, 27)]


def ocean_support(center, t0):
    try:
        sup = support(field, center, t0, R, H)
    except ValueError:
        return None
    sl = tuple(slice(sup[k + "_indices"][0], sup[k + "_indices"][1] + 1) for k in ("time", "latitude", "longitude"))
    if not (np.isfinite(field.u[sl]).all() and np.isfinite(field.v[sl]).all()):
        return None
    return sup


used = [support(field, r["origin_lonlat"], r["start_utc"], R, H) for r in inv["scenarios"]]


def disjoint(sup):
    return not any(intervals_overlap(sup["latitude"], u["latitude"]) and
                   intervals_overlap(sup["longitude"], u["longitude"]) for u in used)


cands = []
lon_axis, lat_axis = np.asarray(field.lon), np.asarray(field.lat)
print("field domain lon", lon_axis.min(), lon_axis.max(), "lat", lat_axis.min(), lat_axis.max())
for lon in np.arange(lon_axis.min() + 0.25, lon_axis.max() - 0.25, 0.75):
    for lat in np.arange(lat_axis.min() + 0.25, lat_axis.max() - 0.25, 0.75):
        c = [float(round(lon, 3)), float(round(lat, 3))]
        sups = [ocean_support(c, t) for t in DATES]
        if any(s is None for s in sups) or not disjoint(sups[0]):
            continue
        fr = LocalFrame(c, R)
        ax = np.linspace(0, R, 9)
        gx, gy = np.meshgrid(ax, ax)
        ll = fr.inverse(np.stack([gx.ravel(), gy.ravel()], -1))
        speed = [np.median(np.hypot(*field.query(t + h, ll[:, 1], ll[:, 0]).T)) for t in DATES for h in (0, H / 2, H)]
        cands.append((float(np.median(speed)), c))
cands.sort()
n = len(cands)
chosen = []
for q in (0.05, 0.25, 0.45, 0.65, 0.85, 0.98):
    target = q * (n - 1)
    for idx in sorted(range(n), key=lambda i: abs(i - target)):
        c = cands[idx][1]
        if all(np.hypot(c[0] - d[0], c[1] - d[1]) >= 1.5 for _, d in chosen):
            chosen.append(cands[idx])
            break
chosen.sort()
print(f"{n} admissible candidate areas; chosen:")
for ai, (spd, c) in enumerate(chosen):
    print(f"  area{ai}: centre {c[0]:.2f}E {c[1]:.2f}N, median speed {spd:.3f} m/s")

rows = [{k: v for k, v in r.items() if k != "support"} for r in inv["scenarios"]]
for ai, (spd, c) in enumerate(chosen):
    for t in DATES:
        day = (t - JAN1) // 86400 + 1
        rows.append(dict(id=f"ood-jan{day:02d}-area{ai}", split="ood_test", origin_lonlat=c, start_utc=t))
spec = OUT / "ood-spec.json"
spec.write_text(json.dumps(dict(config=inv["config"], scenarios=rows,
                                selection_rule=__doc__, candidates=cands), indent=2))
target = OUT / "ood-inventory.json"
if not target.exists():
    subprocess.run([sys.executable, "-m", "peva_sim.build_forcing_inventory", "--field", inv["field_path"],
                    "--spec", str(spec), "--out", str(target)], check=True, cwd=ROOT)
built = json.loads(target.read_text())
print("inventory:", target, "ood scenarios:", sum(r["split"] == "ood_test" for r in built["scenarios"]),
      "audit:", built["audit"])
