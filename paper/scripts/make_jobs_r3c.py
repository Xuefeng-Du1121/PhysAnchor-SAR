"""Campaign C: long-budget MAPPO (10x the standard 102k steps) and OOD for plain MAPPO.
Written to runs/complete-paper-v4/jobs_c.txt as 'name|args' lines for tools/eval_misspec.py."""
import _paths  # noqa: E402  (repo-relative experiment roots)
import os
from pathlib import Path

PR = _paths.root("peva-phys-residual")
OOD = "--scenario-inventory runs/complete-paper-v3/ood/ood-inventory.json --split ood_test --episodes-per-scenario 10"
LM = PR / "runs/complete-paper-v3/train/long-mappo-seed{}"
MAPPO = str(_paths.root("peva-gen-final-v5")) + "/runs/eaai-final-v5b-mappo-seeds0-9/train/mappo-seed{}/checkpoint.pt"
J = []
for s in range(3):                                   # final long-budget policy on test and validation
    c = Path(str(LM).format(s)) / "checkpoint.pt"
    if c.exists():
        J.append(f"test-longmappo-seed{s}|--checkpoint {c}")
        J.append(f"validation-longmappo-seed{s}|--checkpoint {c} --split validation")
for step in (102400, 204800, 409600, 614400, 819200):  # budget curve on validation
    for s in range(3):
        c = Path(str(LM).format(s)) / f"checkpoint-step-{step:09d}.pt"
        if c.exists():
            J.append(f"lc-longmappo-step{step}-seed{s}|--checkpoint {c} --split validation")
for s in range(5):                                   # plain MAPPO out of distribution
    if os.path.exists(MAPPO.format(s)):
        J.append(f"ood-mappo-seed{s}|--checkpoint {MAPPO.format(s)} {OOD}")
for s in range(3):
    c = Path(str(LM).format(s)) / "checkpoint.pt"
    if c.exists():
        J.append(f"ood-longmappo-seed{s}|--checkpoint {c} {OOD}")
out = PR / "runs/complete-paper-v4/jobs_c.txt"
out.write_text("\n".join(J) + "\n")
print(len(J), "jobs ->", out)
