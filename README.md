# aCRW codes

Simulation and figure-generation code accompanying the manuscript **Active Chiral Random Walk**.

The repository contains the Python implementation of the two-dimensional active chiral random walk (aCRW), the continuum chiral active Brownian-particle solver used for comparison, and the scripts used to generate Figures 1--6 of the manuscript. Generated figures and simulation outputs are intentionally not included.

## Repository contents

```text
├── src_python_new/  Python aCRW and cABP implementations
└── figure_scripts/  Figure-generation scripts
```

The figure scripts are:

1. `figure1_trajectories.py` -- representative aCRW trajectories.
2. `figure2_distribution_flux.py` -- probability density and flux.
3. `figure3_msd.py` -- MSD and diagonal/off-diagonal velocity correlations.
4. `figure4_diffusion_phase.py` -- normalized effective-diffusion phase diagram.
5. `figure5_cabp_vacf_msd.py` -- aCRW/cABP MSD and velocity-correlation comparison.
6. `figure6_steady_state_reflective_boundaries.py` -- steady states for three boundary rules.

## Python setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run a figure script from the repository root, for example:

```bash
python figure_scripts/figure5_cabp_vacf_msd.py
```

The scripts create their output directories under `Figures -ACRW/`. Those generated files are excluded from version control.

Some ensemble calculations, especially Figures 3 and 5, are computationally intensive. Their random seeds and ensemble sizes are set near the top of each script.

## Reproducibility notes

- The repository records the source code used for the manuscript, not pre-generated figures.
- Figure scripts write PDF/SVG and JSON metadata files containing the numerical and plotting parameters used in each run.
- The archive is Python-only; the paper figures are produced by the scripts in `figure_scripts/`.
- See `SOURCE_MANIFEST.md` for the provenance and selection of the archived files.

## Citation

Please cite the software using `CITATION.cff`. After the Zenodo deposit is published, add the Zenodo DOI to both `CITATION.cff` and this README. The associated manuscript should also be cited once its bibliographic details are final.

## License

This software is released under the MIT License. See `LICENSE`.
