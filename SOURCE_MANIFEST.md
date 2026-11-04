# Source manifest

Initial archive: 2 November 2026. Manuscript figure update: 4 November 2026.

## Initial source

Python model files were selected from the active development tree
`Desktop/SImulations/ActiveChiralRW-2D/RandomWalk2D.jl/RandomWalk2D.jl (Copy)`.
No Julia sources or generated figures were included.

## Current manuscript figure sources

- Figure 2: the verified specular-boundary replacement, with density notation
  `rho/rho_max`, polarization notation `|m|/|m|_max`, and separated colour-bar
  tick labels. The original flipping metadata describe 75×75 runs; the new
  script regenerates the supplied figure's 150×150 case.
- Figure 3: the expanded transport comparison, including measured even/odd
  coefficients, theoretical stars and one-standard-deviation batch error bars.
- Figure 5: the composite previously called working Figure 18. The original
  composite rendering is retained, while data generation is now portable and
  does not require the original desktop SVGs or caches. The same minmod/SSP RK2
  kinetic solver is archived as `src_python_new/figure5_wall_solver.py`.
- Figure 6: the compact split-time specular/randomized comparison. Coin-wall
  scattering is not one of the figure's selectable conditions.
- Figure 1 and Figure 4: the existing archived scripts are retained.

## Included

- Python source for dynamics, analysis and both model solvers.
- Current entry points for manuscript Figures 1–6 and the Figure 5 helper.
- Tests and documentation explaining numerical and plotting conventions.

## Excluded

- Generated PDF, SVG, PNG, GIF, NPZ, CSV and numerical metadata outputs.
- Python bytecode, Numba caches and virtual environments.
- Manuscript source and bibliography files.
- Historical figure variants and Julia files.

Outputs are written under the repository-level `Figures -ACRW/` directory,
or to an explicitly supplied output directory. The original development
folders and their existing results are not modified by the archived scripts.
