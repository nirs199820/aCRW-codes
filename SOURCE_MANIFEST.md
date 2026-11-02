# Source manifest

This archive was assembled on 2 November 2026 from the active development copy at:

`Desktop/SImulations/ActiveChiralRW-2D/RandomWalk2D.jl/RandomWalk2D.jl (Copy)`

## Included

- `src_python_new/*.py`: current Python model source, including the aCRW dynamics, boundary rules, analysis routines, orientation-resolved distribution solver, and cABP Fokker--Planck solver.
- `figure_scripts/figure1_trajectories.py` through `figure6_steady_state_reflective_boundaries.py`: the scripts corresponding to manuscript Figures 1--6.

## Intentionally excluded

- Generated PDF, SVG, PNG, GIF, JSON, CSV, and NPZ outputs.
- Python bytecode, Numba caches, and temporary files.
- Historical copies such as `cabp_fokker_planck_OLD.py`.
- Scripts for figures outside the present manuscript.
- Manuscript source and bibliography files.
- Julia source and environment files.

The original source tree was copied, not modified. Output paths continue to place generated results under the repository-level `Figures -ACRW/` directory.
