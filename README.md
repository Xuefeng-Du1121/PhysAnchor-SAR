# PEVA-Gen: physics-anchored multi-agent maritime search and rescue

This repository contains only the reusable core model implementations and maritime search-and-rescue simulation environments for UAV swarms.

## Core package

The Python package is under src/peva_sim and includes:

- synthetic and ocean-forced maritime SAR environments;
- ocean-current interpolation, target drift, particle beliefs, and physics value fields;
- communication and confirmation protocols for multi-agent coordination;
- Physics Anchor policies and PEVA-Gen belief/communication models;
- MAPPO, HAPPO, IC3Net, and continuous-action actor/critic model components;
- latent-state, diffusion-belief, transport, replay, and deployment primitives.

The public package entry point exposes Config and MaritimeSAR from peva_sim.env. Install the package with pip install -e . after installing requirements.txt.

## Release scope

Training entry points, evaluation programs, experiment launchers, scenario/configuration inventories, tests, plotting scripts, generated results, and checkpoints have been removed from this release. The remaining source is intended as the core model and simulation layer for integration into a separate training or evaluation system.

## License

MIT; see LICENSE.
