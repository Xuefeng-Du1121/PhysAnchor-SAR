"""Collect per-episode detection times and the first test episode's trajectories
from the official held-out test traces (no re-evaluation, no new metric).

For each method and seed we store, per episode, the detection time of each
survivor (inf if never found). Success at t=T must equal summary.json; this is
checked and printed.
"""
import _paths  # noqa: E402  (repo-relative experiment roots)
import glob, json, os, sys
from pathlib import Path
import numpy as np

PR = str(_paths.root("peva-phys-residual"))
V5 = str(_paths.root("peva-gen-final-v5"))
V1 = PR + "/runs/campaign-v1/eval/test"
CP = PR + "/runs/complete-paper-v1/test"
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
DT, T = 30.0, 10800.0


def seed_dirs(patterns):
    out = {}
    for pat in patterns:
        for d in sorted(glob.glob(pat)):
            s = d.rsplit("-seed", 1)[-1]
            if s not in out and os.path.exists(d + "/summary.json"):
                out[s] = d
    return out


def det_times(npz):
    z = np.load(npz)
    found, ts = z["found"], z["time_s"]
    times = np.full(found.shape[1], np.inf)
    for g in range(found.shape[1]):
        hit = np.flatnonzero(found[:, g])
        if len(hit):
            times[g] = ts[hit[0]]
        elif z["terminal_found"][g]:
            times[g] = ts[-1] + DT
    return times


LEARNED = {
    "anchor30": [f"{V1}/anchor-ang30-seed*", f"{CP}/anchor-ang30-seed*"],
    "anchor15": [f"{V1}/anchor-mappo-seed*", f"{CP}/anchor-mappo-seed*"],
    "anchor0": [f"{V1}/anchor-ang0-seed*", f"{CP}/anchor-ang0-seed*"],
    "ic3net": [f"{V1}/ic3net-seed*"],
    "pvfmappo": [f"{V5}/runs/campaign-v3/test/pvf-mappo-seed*"],
    "mappo": [f"{V5}/runs/campaign-v3/test/mappo-seed*"],
    "pevagen": [f"{V5}/runs/peva-test-eval-v2/eval/peva-gen-seed*"],
}
data, first_ep = {}, {}
for name, pats in LEARNED.items():
    per_seed = []
    for s, d in sorted(seed_dirs(pats).items(), key=lambda kv: int(kv[0])):
        files = sorted(glob.glob(d + "/episode-*.npz"))
        tt = np.stack([det_times(f) for f in files])
        summ = json.load(open(d + "/summary.json"))
        sr = summ["success_rate"]["mean"] if isinstance(summ["success_rate"], dict) else summ["success_rate"]
        mine = float(np.mean(np.all(np.isfinite(tt), axis=1)))
        # summary success_rate is the mean fraction found per episode (info["success_rate"])
        frac = float(np.mean(np.isfinite(tt)))
        per_seed.append(tt)
        print(f"{name:9s} seed {s:>2s}: episodes={len(files)} all-found={mine:.3f} "
              f"found-fraction={frac:.3f} summary={sr:.3f}")
        if name not in first_ep:
            z = np.load(files[0])
            first_ep[name] = dict(uavs=z["uavs_m"], targets=z["targets_m"], found=z["found"],
                                  t=z["time_s"], seed=s)
    data[name] = np.stack(per_seed)          # seeds x episodes x targets

cl = {}
for f in sorted(glob.glob(f"{CP}/classical/*-episode-*.npz")):
    m, idx = os.path.basename(f).rsplit("-episode-", 1)
    cl.setdefault(m, []).append((int(idx[:-4]), f))
summ = json.load(open(f"{CP}/classical/summary.json"))["summaries"]
for m, rows in cl.items():
    rows.sort()
    tt = np.stack([det_times(f) for _, f in rows])
    print(f"{m:26s} episodes={len(rows)} found-fraction={np.mean(np.isfinite(tt)):.3f} "
          f"summary={summ[m]['success_rate']['mean']:.3f}")
    data["cl_" + m] = tt[None]
    z = np.load(rows[0][1])
    first_ep["cl_" + m] = dict(uavs=z["uavs_m"], targets=z["targets_m"], found=z["found"], t=z["time_s"], seed="-")

flat = {}
for k, v in data.items():
    flat["det_" + k] = v
for k, v in first_ep.items():
    for kk, vv in v.items():
        flat[f"ep0_{k}_{kk}"] = np.asarray(vv)
np.savez_compressed(OUT / "test_traces.npz", **flat)
print("DONE", sorted(data))
