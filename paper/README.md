# Paper scripts

* `scripts/` – evaluation drivers used for the paper. `eval_rec.py` evaluates learned checkpoints,
  the network-free physics anchor (`--physics-only`) and non-learning planners (`--classical`)
  on identical scenarios and seeds, with optional stress and misspecification settings.
  `extra_planners.py` holds the planners not in `peva_sim.classical_policies`.
  `bc_control.py` is the behavior-cloning control, and `build_ood.py` selects the 24 OOD scenarios
  (its output is in `configs/ood/`).
* `scripts/fig3/` – `results_r3.py` collects every summary into one JSON file. `table_r3.py`,
  `numbers_r3.py` and the `fig_*.py` scripts turn it into the tables, the LaTeX number macros
  and the figures.
* `scripts/viz/` – recorders of the trajectories and fields drawn in the qualitative figures.
* `campaigns/` – the shell drivers that ran the training and evaluation campaigns. Set
  `PEVA_REPO`, `PEVA_DATA_ROOT` and `PY` before using them.

Result files and checkpoints are not part of this release (see the top-level README).
