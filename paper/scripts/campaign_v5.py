"""campaign-v5 (revision round 2). Writes job lists; the shell driver runs them.

eval jobs (CPU, eval_misspec.py, all with frozen checkpoints/planners):
  M  mis-specified anchor, test split: true G in {2,5}, anchor peaks K=3 (and K=G)
  Z  zero-residual angle sweep theta in {0..90} on validation and test
  O  supplementary held-out scenarios (24 = 6 sea areas x 4 dates), 10 episodes each
  S  shaping-only on test
  P  planner parameter sensitivity on validation (travel scale x exclusion radius)
train jobs (GPU): shaping-only seeds 3-9, BC-initialized control, 1M-step baselines.
"""
import _paths  # noqa: E402  (repo-relative experiment roots)
import json
from pathlib import Path

PR = str(_paths.root("peva-phys-residual"))
R = "runs/complete-paper-v3"
CK = json.load(open(f"{PR}/{R}/ckpt-map.json"))
ZR = "runs/complete-paper-v2/train/anchor-zero-residual/checkpoint.pt"
OOD = f"{R}/ood/ood-inventory.json"
TWO = [f"runs/complete-paper-v2/train/anchor-2upd-seed{s}/checkpoint-step-000000128.pt" for s in range(10)]
ev = []


def job(name, *args):
    ev.append(f"{name}\t" + " ".join(["tools/eval_misspec.py", *map(str, args), "--out", f"{R}/eval/{name}"]))


# M: mis-specified anchor
for G in (2, 5):
    for K in (3, G):
        tag = f"G{G}K{K}"
        for s, row in sorted(CK["anchor-ang30"].items(), key=lambda kv: int(kv[0])):
            job(f"mis-{tag}-ang30-seed{s}", "--checkpoint", row["checkpoint"], "--true-targets", G,
                "--anchor-peaks", K, "--split", "test", "--episodes-per-scenario", 50)
        job(f"mis-{tag}-zero30", "--checkpoint", ZR, "--zero-residual", "--angle", 30, "--true-targets", G,
            "--anchor-peaks", K, "--split", "test", "--episodes-per-scenario", 50)
    for m in ("pvf-greedy", "cbba", "oracle-target"):
        job(f"mis-G{G}-{m}", "--classical", m, "--true-targets", G, "--split", "test", "--episodes-per-scenario", 50)
# Z: angle sweep of the pure physics policy
for split in ("validation", "test"):
    for th in (0, 15, 30, 45, 60, 75, 90):
        job(f"zero-{split}-ang{th}", "--checkpoint", ZR, "--zero-residual", "--angle", th, "--split", split,
            "--episodes-per-scenario", 50)
# O: supplementary held-out scenarios
O = ["--scenario-inventory", OOD, "--split", "ood_test", "--episodes-per-scenario", 10]
for group in ("anchor-ang30", "anchor-mappo", "pvf-mappo", "ic3net", "shape1-noanchor"):
    for s, row in CK[group].items():
        job(f"ood-{group}-seed{s}", "--checkpoint", row["checkpoint"], *O)
for s, p in enumerate(TWO):
    job(f"ood-anchor-2upd-seed{s}", "--checkpoint", p, *O)
for th in (15, 30):
    job(f"ood-zero{th}", "--checkpoint", ZR, "--zero-residual", "--angle", th, *O)
for m in ("pvf-greedy", "cbba", "lawnmower-coverage", "bayesian-information-gain", "greedy-probability", "oracle-target"):
    job(f"ood-{m}", "--classical", m, *O)
# S: shaping-only (existing seeds) on test
for s, row in CK["shape1-noanchor"].items():
    job(f"test-shape1-noanchor-seed{s}", "--checkpoint", row["checkpoint"], "--split", "test",
        "--episodes-per-scenario", 50)
# P: planner sensitivity on validation, then the selected setting is run on test by the driver
for m in ("pvf-greedy", "cbba"):
    for tr in (1250, 2500, 5000, 10000, 20000, 1e9):
        for ex in (0.5, 1.0, 2.0):
            job(f"plan-validation-{m}-tr{int(min(tr, 1e9))}-ex{ex}", "--classical", m, "--travel-scale-m", tr,
                "--exclusion-mult", ex, "--split", "validation", "--episodes-per-scenario", 50)

Path(f"{PR}/{R}/eval-jobs.tsv").write_text("\n".join(ev) + "\n")
print(len(ev), "eval jobs")
