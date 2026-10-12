# Shared environment for the physics-residual study
export PEVA_DATA_ROOT=${PEVA_DATA_ROOT:?set PEVA_DATA_ROOT}
export PYTHONPATH=${PEVA_REPO:?set PEVA_REPO to this repository}/src
export PY=${PY:-python}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
