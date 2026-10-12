# Data provenance and usage

The simulator is driven by historical ocean forcing obtained from the Copernicus Marine Service. The exact product identifier, geographic bounding box, time range, download date, and preprocessing checksum used for the paper must be recorded in the companion data repository and in the frozen inventory file.

## Separate data repository

The companion public data repository is:

`https://github.com/Xuefeng-Du1121/Peva-Gen-data`

The data repository should contain:

1. a `README.md` with the Copernicus product URL and license/attribution;
2. a `MANIFEST.json` listing every downloaded file, SHA-256 checksum, dimensions, timestamps, and split assignment;
3. preprocessing scripts or notebooks sufficient to regenerate the compact forcing files;
4. a small public sample for smoke tests;
5. instructions for obtaining any files that cannot be redistributed directly.

Do not commit credentials, Copernicus access tokens, private server paths, or raw files that the upstream license does not permit redistribution.

## Expected local layout

Set `PEVA_DATA_ROOT` to the `data` directory of the extracted bundle, i.e. the directory that
contains `raw/environment/copernicus/multobs_currents_east_china_2018_01.nc`. The frozen inventory
(`configs/formal-ocean-forcing-inventory-v1.json`) records that file by its absolute path on the
machine that froze it. When the path does not exist, the loader finds the same relative path below
`PEVA_DATA_ROOT` and checks its SHA-256 against the inventory before using it.

## Reproducibility record

For every public result, archive the following alongside the result table:

- Copernicus product identifier and source URL;
- download date and preprocessing version;
- data manifest SHA-256;
- split and scenario identifiers;
- code commit SHA;
- random seeds;
- PyTorch/CUDA versions.

The paper's current evidence is simulation based, even though the forcing is derived from historical ocean data. It should not be described as a real-UAV or field deployment validation.
