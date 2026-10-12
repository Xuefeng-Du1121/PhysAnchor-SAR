# Physics-anchored MARL for maritime search and rescue

This repository contains the core PEVA-Gen and physics-anchored multi-agent reinforcement learning implementation for maritime search and rescue with UAV swarms.

## Contents

- src/peva_sim/: simulator, ocean forcing, particle belief, physics value fields, policies, learners, and training/evaluation entry points.
- configs/: frozen training, validation, test, and OOD scenario inventories.
- paper/scripts/: experiment and evaluation drivers.
- paper/campaigns/: training and evaluation campaign launchers.
- tests/: unit tests.

## Installation

Use Python 3.10 or 3.11, install the dependencies from requirements.txt, install the package with pip install -e ., and run pytest -q. The Copernicus surface-current data path is configured with PEVA_DATA_ROOT; see DATA.md.

## Training and evaluation

The main entry points are the train_formal_* and evaluate_formal_* modules under src/peva_sim. The scripts under paper/scripts run reproducible evaluations for learned policies, physics-only policies, classical planners, and stress settings.

## Release scope

This release intentionally omits all paper figure-generation, visualization, and result-presentation Python scripts, as well as generated figures, checkpoints, and per-episode result files.

## License

MIT; see LICENSE.
