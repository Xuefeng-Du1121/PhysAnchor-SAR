"""Regional Copernicus current field at the start time of given scenarios (raw grid, land = NaN)."""
import _paths  # noqa: E402  (repo-relative experiment roots)
import json
import sys
from pathlib import Path
import numpy as np
from peva_sim.ocean_field import OceanField

ROOT = _paths.root("peva-phys-residual")
inv = json.loads((ROOT / "configs/formal-ocean-forcing-inventory-v1.json").read_text())
ood = json.loads((ROOT / "runs/complete-paper-v3/ood/ood-inventory.json").read_text())
rows = {r["id"]: r for r in inv["scenarios"] + ood["scenarios"]}
f = OceanField(inv["field_path"])
out = Path(sys.argv[1])
for sid in sys.argv[2:]:
    it = int(np.argmin(np.abs(f.time - rows[sid]["start_utc"])))
    it3 = int(np.argmin(np.abs(f.time - rows[sid]["start_utc"] - 10800)))
    np.savez_compressed(out / f"region_{sid}.npz", lon=np.asarray(f.lon), lat=np.asarray(f.lat),
                        u=f.u[it], v=f.v[it], u3=f.u[it3], v3=f.v[it3], origin=np.asarray(rows[sid]["origin_lonlat"]),
                        start_utc=rows[sid]["start_utc"])
    print("saved", sid)
