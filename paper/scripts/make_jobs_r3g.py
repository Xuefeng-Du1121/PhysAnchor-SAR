"""Campaign G: learning-curve checkpoints on validation and the alpha sweep of the network-free policy at
theta = 15 deg (to match the learned strength ablation). Written to runs/complete-paper-v4/jobs_g.txt."""
import _paths  # noqa: E402  (repo-relative experiment roots)
from pathlib import Path

PR = _paths.root("peva-phys-residual")
V5 = (_paths.root("peva-gen-final-v5") / "runs")
STEPS = (25600, 51200, 102400, 153600)
J = []


def ck(d, step):
    p = Path(d) / f"checkpoint-step-{step:09d}.pt"
    return p if p.exists() else None


for s in range(3):                                       # MAPPO, early part of the curve (long run covers >=102k)
    for st in (25600, 51200):
        c = ck(V5 / f"eaai-final-v5b-mappo-seeds0-9/train/mappo-seed{s}", st)
        if c:
            J.append(f"lc-mappo-step{st}-seed{s}|--checkpoint {c} --split validation")
for s in (3, 4, 5):                                      # shaping only, no anchor
    for st in STEPS:
        c = ck(PR / f"runs/complete-paper-v3/train/shape1-noanchor-seed{s}", st)
        if c:
            J.append(f"lc-shapeonly-step{st}-seed{s}|--checkpoint {c} --split validation")
for s in range(3):                                       # anchored policy, theta = 30 deg
    for root in ("complete-paper-v1", "campaign-v1"):
        d = PR / f"runs/{root}/train/anchor-ang30-seed{s}"
        if d.exists():
            for st in STEPS:
                c = ck(d, st)
                if c:
                    J.append(f"lc-anchor30-step{st}-seed{s}|--checkpoint {c} --split validation")
            break
for split in ("validation", "test"):                     # network-free policy, alpha sweep at theta = 15 deg
    sp = "" if split == "test" else "--split validation"
    for al in (1, 2, 3, 4, 6, 8, 12):
        J.append(f"sens15-{split}-alpha{al}|--physics-only --angle 15 --alpha {al} {sp}")
out = PR / "runs/complete-paper-v4/jobs_g.txt"
out.write_text("\n".join(J) + "\n")
print(len(J), "jobs ->", out)
