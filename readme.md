# Drp1 Mitochondrial Fission Model

Accompanying code to the paper "A dynamical systems model for the total fission
rate in Drp1-dependent mitochondrial fission"


## Software environment

If you are running [nix](https://nixos.org) setting up a software environment
is easy: `nix develop` will drop you in a shell with all the dependencies ready
to go. If you are not running nix there is a `pyproject.toml` which should
allow you to `pip install -e .` to get all the python dependencies sorted.

## Running

### Global Sensitivity Analysis

The repository ships example output from running the global sensitivity
analysis with a seed of 800. To repeat the analysis there are three different
commands for the three different analyses.

- `python -m GSA_dimensionless`
- `python -m GSA_4params`
- `python -m GSA_5params`

Each of these respect the following environment variables:

- `NSLOTS` how many cores to use
- `SEED` what seed to use
- `scratch` where to save intermediate data.

### Figures

`make svg` will build svg versions of all the figures in the paper. `make all`
will build the versions of the figures in the paper. This will require extra
dependencies to convert the svgs to tiff files. The nix environment contains
all these needed dependencies.
