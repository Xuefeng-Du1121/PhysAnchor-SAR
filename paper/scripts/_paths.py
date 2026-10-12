"""Roots of the result trees the paper scripts read and write.

The scripts were written against two experiment trees, the physics-residual study and the earlier
formal-baseline tree (MAPPO, CommNet, PVF-MAPPO, PEVA-Gen). Both default to the repository root,
so configs/ and runs/<run-name>/ are found there. Override with
PEVA_PAPER_ROOT (the physics-residual tree) and PEVA_BASELINE_ROOT (the baseline tree), e.g. to
write new runs somewhere else.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PR = Path(os.environ.get("PEVA_PAPER_ROOT", REPO))
V5 = Path(os.environ.get("PEVA_BASELINE_ROOT", REPO))
EXTERNAL = Path(os.environ.get("PEVA_EXTERNAL_ROOT", REPO / "runs" / "external"))


def root(name):
    """Root of a named experiment tree."""
    return {"peva-phys-residual": PR, "peva-gen-final-v5": V5}.get(name, EXTERNAL / name)
