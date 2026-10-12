"""Map every evaluation used in the paper to the checkpoint file it evaluated (by SHA-256)."""
import _paths  # noqa: E402  (repo-relative experiment roots)
import glob
import hashlib
import json
import os
import sys

PR = str(_paths.root("peva-phys-residual"))
V5 = str(_paths.root("peva-gen-final-v5"))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


cands = []
for root in (PR + "/runs", V5 + "/runs"):
    for name in ("checkpoint.pt", "checkpoint-step-000204800.pt"):
        cands += glob.glob(f"{root}/**/{name}", recursive=True)
by_sha = {}
for c in cands:
    by_sha.setdefault(sha(c), c)
print("hashed", len(cands), "candidate checkpoints", flush=True)

GROUPS = {
    "anchor-mappo": [f"{PR}/runs/campaign-v1/eval/test/anchor-mappo-seed*", f"{PR}/runs/complete-paper-v1/test/anchor-mappo-seed*"],
    "anchor-ang30": [f"{PR}/runs/campaign-v1/eval/test/anchor-ang30-seed*", f"{PR}/runs/complete-paper-v1/test/anchor-ang30-seed*"],
    "ic3net": [f"{PR}/runs/campaign-v1/eval/test/ic3net-seed*"],
    "pvf-mappo": [f"{V5}/runs/campaign-v3/test/pvf-mappo-seed*"],
    "shape1-noanchor": [f"{PR}/runs/stage3-v1/validation/shape1-noanchor-seed*"],
}
out = {}
for group, patterns in GROUPS.items():
    out[group] = {}
    for pat in patterns:
        for d in sorted(glob.glob(pat)):
            seed = d.rsplit("-seed", 1)[-1]
            if seed in out[group] or not os.path.exists(d + "/summary.json"):
                continue
            s = json.load(open(d + "/summary.json"))
            h = s.get("checkpoint_sha256")
            path = by_sha.get(h)
            out[group][seed] = dict(checkpoint=path, sha=h, summary=d + "/summary.json",
                                    success_rate=(s["success_rate"]["mean"] if isinstance(s["success_rate"], dict)
                                                  else s["success_rate"]))
            print(group, seed, "->", path, flush=True)
missing = [(g, s) for g, v in out.items() for s, r in v.items() if r["checkpoint"] is None]
json.dump(out, open(sys.argv[1], "w"), indent=2)
print("missing:", missing)
