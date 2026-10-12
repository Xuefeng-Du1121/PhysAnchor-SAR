"""Job list for revision-3 campaign B (all evaluation, no training). Written to runs/complete-paper-v4/jobs_b.txt
as 'name|args' lines for tools/eval_misspec.py. Order = priority."""
import _paths  # noqa: E402  (repo-relative experiment roots)
import glob
import json
import os
from pathlib import Path

PR = _paths.root("peva-phys-residual")
CK = json.load(open(PR / "runs/complete-paper-v3/ckpt-map.json"))
OOD = "--scenario-inventory runs/complete-paper-v3/ood/ood-inventory.json --split ood_test --episodes-per-scenario 10"
J = []


def add(name, args):
    J.append(f"{name}|{args}")


def ck(group, seed):
    return CK[group][str(seed)]["checkpoint"]


ippo = str(_paths.root("peva-gen-perf-dev")) + "/runs/eaai-final-v4-ippo-seeds0-9/train/ippo-seed{}/checkpoint.pt"
mappo = sorted(glob.glob(str(_paths.root("peva-gen-final-v5")) + "/runs/eaai-final-v5b-mappo-seeds0-9/train/mappo-seed{}/checkpoint*.pt".format(0)))
PLANNERS = ["sector-search", "expanding-square", "aco-search", "pso-search", "mpc-horizon", "pvf-greedy", "cbba",
            "bayesian-information-gain", "lawnmower-coverage", "voronoi-coverage", "levy-random-walk"]

# 1. trajectories of one test scenario for the qualitative figures (cheap, first)
for p in ("sector-search", "expanding-square", "aco-search", "pvf-greedy", "cbba", "mpc-horizon"):
    add(f"traj-{p}", f"--classical {p} --episodes-per-scenario 1 --record-episodes 1")
add("traj-phys30", "--physics-only --angle 30 --episodes-per-scenario 1 --record-episodes 1")
add("traj-anchor30-s3", f"--checkpoint {ck('anchor-ang30', 3)} --episodes-per-scenario 1 --record-episodes 1")
add("traj-ic3net-s0", f"--checkpoint {ck('ic3net', 0)} --episodes-per-scenario 1 --record-episodes 1")
add("traj-pvfmappo-s0", f"--checkpoint {ck('pvf-mappo', 0)} --episodes-per-scenario 1 --record-episodes 1")
add("traj-ippo-s0", f"--checkpoint {ippo.format(0)} --episodes-per-scenario 1 --record-episodes 1")

# 2. IPPO (independent learners) on test, 10 seeds; and validation through the same evaluator
for s in range(10):
    if os.path.exists(ippo.format(s)):
        add(f"test-ippo-seed{s}", f"--checkpoint {ippo.format(s)}")
# 3. shaping-only seeds 3-9 and the behaviour-cloning control on test and validation
for s in range(3, 10):
    c = PR / f"runs/complete-paper-v3/train/shape1-noanchor-seed{s}/checkpoint.pt"
    if c.exists():
        add(f"test-shape1-noanchor-seed{s}", f"--checkpoint {c}")
        add(f"validation-shape1-noanchor-seed{s}", f"--checkpoint {c} --split validation")
for s in range(3):
    c = PR / f"runs/complete-paper-v3/train/bc-finetune-seed{s}/checkpoint.pt"
    if c.exists():
        add(f"test-bcft-seed{s}", f"--checkpoint {c}")
        add(f"validation-bcft-seed{s}", f"--checkpoint {c} --split validation")
    c0 = PR / f"runs/complete-paper-v3/train/bc-init-seed{s}.pt"
    if c0.exists():
        add(f"test-bcinit-seed{s}", f"--checkpoint {c0}")

# 4. operational stress on the TEST split for the full method set
STRESS = {"det06": "--detection-p 0.6", "drift03": "--drift-error-bound-mps 0.3",
          "drift05": "--drift-error-bound-mps 0.5", "drop05": "--packet-drop 0.5",
          "hard": "--detection-p 0.6 --packet-drop 0.5 --drift-error-bound-mps 0.3"}
for cond, flags in STRESS.items():
    add(f"stress-{cond}-phys30", f"--physics-only --angle 30 {flags}")
    for s in range(3):
        add(f"stress-{cond}-anchor30-seed{s}", f"--checkpoint {ck('anchor-ang30', s)} {flags}")
    for p in PLANNERS[:9]:
        add(f"stress-{cond}-{p}", f"--classical {p} {flags}")
    add(f"stress-{cond}-ic3net-seed0", f"--checkpoint {ck('ic3net', 0)} {flags}")
    add(f"stress-{cond}-pvfmappo-seed0", f"--checkpoint {ck('pvf-mappo', 0)} {flags}")
    add(f"stress-{cond}-ippo-seed0", f"--checkpoint {ippo.format(0)} {flags}")

# 5. number of survivors G in {1,2,3,4,5,6} on test (G=2,5 learned already in campaign-v5)
for G in (1, 2, 4, 5, 6):
    add(f"count-G{G}-phys30", f"--physics-only --angle 30 --true-targets {G} --anchor-peaks {G}")
    for p in PLANNERS[:8]:
        add(f"count-G{G}-{p}", f"--classical {p} --true-targets {G}")
for G in (1, 4, 6):
    for s in range(3):
        add(f"count-G{G}-anchor30-seed{s}", f"--checkpoint {ck('anchor-ang30', s)} --true-targets {G} --anchor-peaks {G}")
    add(f"count-G{G}-ic3net-seed0", f"--checkpoint {ck('ic3net', 0)} --true-targets {G}")

# 6. team size N in {2, 12} to complete the curve (3,4,8,10 in campaign A)
for n in (2, 12):
    add(f"team{n}-phys30", f"--physics-only --angle 30 --n-uavs {n}")
    for p in ("pvf-greedy", "cbba", "expanding-square", "sector-search", "lawnmower-coverage", "aco-search"):
        add(f"team{n}-{p}", f"--classical {p} --n-uavs {n}")
for n in (3, 4, 8, 10):
    add(f"team{n}-aco-search", f"--classical aco-search --n-uavs {n}")

# 7. finer sensitivity of the pure physics policy (theta, alpha) on validation and test
for split in ("validation", "test"):
    sp = "" if split == "test" else "--split validation"
    for th in (5, 10, 20, 25, 35, 40, 50):
        add(f"sens-{split}-theta{th}", f"--physics-only --angle {th} {sp}")
    for al in (1, 2, 3, 6, 8, 12):
        add(f"sens-{split}-alpha{al}", f"--physics-only --angle 30 --alpha {al} {sp}")

# 8. OOD for IPPO and the two learned residual controls
for s in range(5):
    if os.path.exists(ippo.format(s)):
        add(f"ood-ippo-seed{s}", f"--checkpoint {ippo.format(s)} {OOD}")
for p in ("pvf-greedy", "cbba", "bayesian-information-gain", "lawnmower-coverage"):
    add(f"ood-{p}", f"--classical {p} {OOD}")

out = PR / "runs/complete-paper-v4/jobs_b.txt"
out.write_text("\n".join(J) + "\n")
print(len(J), "jobs ->", out)
