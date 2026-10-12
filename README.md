# Physics-anchored MARL for maritime search and rescue

Code for the paper *Writing the Value Instead of Learning It: Physics-Anchored Multi-Agent
Reinforcement Learning for Maritime Search and Rescue with Unmanned Aerial Vehicle Swarms*.

A swarm of UAVs searches for survivors drifting in historical Copernicus surface currents. Each UAV
computes a physics value field from a shared particle belief. A fixed, parameter-free **physics
anchor** reduces the field to one peak per expected survivor and turns it into a flight direction
for each UAV, rotated by a fixed role angle θ. A recurrent MAPPO actor learns only a **residual**
on top of the anchor. The same code base contains every learned and non-learning baseline of the paper.

> **Release status.** This branch contains the code, the frozen scenario inventories and the
> experiment and analysis scripts. Trained checkpoints and per-seed / per-episode result files are
> not included. They will be released after the paper is accepted.

## Layout

| Path | Content |
|---|---|
| `src/peva_sim/` | simulator, ocean forcing, particle belief, physics value field, learners and planners |
| `src/peva_sim/formal_mappo.py`, `train_formal_mappo.py` | MAPPO actor/critic with the optional physics anchor (`--pvf-anchor-strength`, `--pvf-anchor-angle-deg`) and optional potential-based physics shaping (`--phys-shaping-coef`) |
| `src/peva_sim/formal_happo.py`, `formal_ic3net.py`, `formal_offpolicy.py` | HAPPO, IC3Net, MADDPG / MATD3 / MASAC baselines (anchor optional for the off-policy actors) |
| `src/peva_sim/peva_gen.py`, `formal_peva.py` | PEVA-Gen baseline (value-governed diffusion belief, communication bottleneck, replay) and its ablations |
| `src/peva_sim/classical_policies.py` | PVF-greedy, CBBA and the other belief-driven planners |
| `configs/formal-ocean-forcing-inventory-v1.json` | frozen training / validation / test scenario inventory (SHA-256 checked at load time) |
| `configs/ood/` | specification and inventory of the 24 out-of-distribution scenarios |
| `paper/scripts/` | evaluation drivers used for the paper (anchor-only policy, misspecified anchors, classical planners, stress tests, behavior-cloning control, extra planners) |
| `paper/scripts/fig3/` | scripts that build every table, figure and reported number of the paper from the result files |
| `paper/scripts/viz/` | recorders for the trajectory and field data shown in the qualitative figures |
| `paper/campaigns/` | the shell drivers of the training / evaluation campaigns, with machine paths replaced by variables |
| `tests/` | unit tests (`pytest -q`) |

## Installation

Python 3.10 or 3.11.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # install the CUDA build of PyTorch first if you train on GPU
pip install -e .
pytest -q
```

## Data

The only external input is the Copernicus Marine surface-current subset (variables `total_u`,
`total_v`, 117.625–135.375°E, 19.625–32.375°N, 0.25°, hourly, 1 January – 1 February 2018).
It is distributed in the companion data repository (see `DATA.md`). Point `PEVA_DATA_ROOT` at the
`data` directory of the bundle:

```bash
export PEVA_DATA_ROOT=/path/to/ocean_system_data_20260913T024902Z/drift-trajectory-system/data_platform/data
```

The frozen inventory records the forcing file by its path on the machine that froze it. If that
path does not exist, the loader looks the file up below `PEVA_DATA_ROOT` and then verifies its
SHA-256 (`2f9b6572…`), so a modified or different file is rejected.

## Training the anchored policy

The paper's anchored policy (α = 4, θ = 30°; 800 PPO updates of 256 steps, 10 seeds):

```bash
python -m peva_sim.train_formal_mappo \
  --inventory configs/formal-ocean-forcing-inventory-v1.json --out runs/anchor30-seed0 --seed 0 \
  --updates 800 --rollout 256 --epochs 4 --minibatch 64 --recurrent-chunk 16 \
  --hidden-dim 128 --gae-lambda 0.95 --entropy-coef 0.01 --snapshot-every 50 \
  --device cuda --pvf --pvf-anchor-strength 4 --pvf-anchor-angle-deg 30
```

Leave out `--pvf-anchor-strength` for plain MAPPO, and leave out `--pvf` as well for MAPPO without
the physics field as input. The other learners have their own `train_formal_*` entry points;
`run_formal_study` lists every method name and its flags.

## Evaluation

```bash
# learned policy, held-out test dates (2 scenarios x 50 episodes)
python -m peva_sim.evaluate_formal_mappo --inventory configs/formal-ocean-forcing-inventory-v1.json \
  --checkpoint runs/anchor30-seed0/checkpoint.pt --out runs/eval/test/anchor30-seed0 \
  --split test --episodes-per-scenario 50

# network-free physics anchor, and a non-learning planner, with the same scenarios and seeds
export PYTHONPATH=$PWD/src
python paper/scripts/eval_rec.py --physics-only --alpha 4 --angle 30 --split test --out runs/eval/test/phys30
python paper/scripts/eval_rec.py --classical sector-search --split test --out runs/eval/test/sector-search
```

For the OOD scenarios add `--split ood_test --scenario-inventory configs/ood/ood-inventory.json`.
`paper/scripts/eval_rec.py --help` lists the stress and misspecification options (detection
probability, packet loss, drift error, number of survivors and anchor peaks, angle override).

## Reproducing the tables and figures

`paper/scripts/` reads two experiment trees: the physics-residual study and the formal-baseline
tree. Their roots default to the repository root and can be set with `PEVA_PAPER_ROOT` and
`PEVA_BASELINE_ROOT` (see `paper/scripts/_paths.py`). `paper/campaigns/` shows the run names the
analysis scripts expect. Once the result files exist:

```bash
cd paper/scripts/fig3
python results_r3.py            # collect all summaries -> results_r3.json
python table_r3.py              # Tables 2 and 5, paired statistics
python numbers_r3.py            # every number quoted in the text, as LaTeX macros
python fig_r3_intro.py; python fig_r3_pipeline.py; python fig_r3_study.py; python fig_r3_mission.py
python fig_r3_results.py ttd written
python fig_r4.py curves ablation gen robust count
```

Every plotted or quoted value is read from `summary.json` / `episodes.jsonl` files. None is typed in.

## License

Code: MIT (see `LICENSE`). The ocean data keep the license and attribution terms of the
Copernicus Marine Service.
