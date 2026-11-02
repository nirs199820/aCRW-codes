"""Figure 1: 2x2 trajectory panels for four ACRW walker types."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb

import importlib.util

ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT
sys.path.insert(0, str(ROOT))


def _load_module(name: str, path: Path):
  """Load a module file without executing package __init__ (avoids numba)."""
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load module from {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


_src = ROOT / "src_python_new"
_directions = _load_module("src_python_new.directions", _src / "directions.py")
_states = _load_module("src_python_new.states", _src / "states.py")
_coins = _load_module("src_python_new.coins", _src / "coins.py")
_percolation = _load_module("src_python_new.percolation", _src / "percolation.py")

DIRECTION_TABLE = _coins.DIRECTION_TABLE
sample_coin_direction_batch_from_coin = _coins.sample_coin_direction_batch_from_coin
make_general_coin = _percolation.make_general_coin

T = 300
GRID_SIZE = 30
SEED =  520476 #np.random.randint(0, 1000000) #520476, 667433, 825810, 847635, 492418, 644750, 749297
print(f"SEED: {SEED}")
SAVE_DIR = REPO_ROOT / "Figures -ACRW/Fig1 - MSD and Trajectories"
OUTPUT_PDF = SAVE_DIR / "figure1_trajectories.pdf"
OUTPUT_SVG = SAVE_DIR / "figure1_trajectories.svg"
OUTPUT_META = SAVE_DIR / "figure1_meta.json"

WALKERS = [
  ("Random", 0.0, 0.0),
  ("Anti-persistent", -0.5, 0.2),
  ("Chiral", 0.0, 0.5),
  ("Active Chiral", 0.5, 0.2),
]

# Palette matched to figure1_msd / reference figure (red / purple / blue / green)
COLORS = ["#D21517", "#972B91", "#3F70AD", "#32A769"]
COLOR_PALE_FRACTION = 0.3  # blend early segments toward white

_TIMES_PATH = Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf")
if _TIMES_PATH.exists():
  font_manager.fontManager.addfont(str(_TIMES_PATH))

plt.rcParams.update({
  "font.family": "serif",
  "font.serif": ["Times New Roman", "Times", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
  "mathtext.fontset": "stix",
  "mathtext.default": "regular",
  "axes.titlesize": 26,
  "axes.titlepad": 12,
  "axes.labelsize": 22,
  "xtick.labelsize": 20,
  "ytick.labelsize": 20,
  "axes.linewidth": 2.5,
})


def simulate_single_trajectory(
  coin: np.ndarray,
  max_steps: int,
  *,
  x0: int = 0,
  y0: int = 0,
  seed: int,
  grid_size: int = GRID_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
  """Simulate one trajectory on a periodic square grid."""
  rng = np.random.default_rng(seed)
  dir_idx = np.array([rng.integers(0, 4)], dtype=np.int32)

  x = x0 % grid_size
  y = y0 % grid_size
  traj_x = np.zeros(max_steps + 1, dtype=np.int32)
  traj_y = np.zeros(max_steps + 1, dtype=np.int32)
  traj_x[0] = x
  traj_y[0] = y

  for step in range(1, max_steps + 1):
    dir_idx = sample_coin_direction_batch_from_coin(coin, dir_idx, rng)
    x = (x + int(DIRECTION_TABLE[dir_idx[0], 0])) % grid_size
    y = (y + int(DIRECTION_TABLE[dir_idx[0], 1])) % grid_size
    traj_x[step] = x
    traj_y[step] = y

  return traj_x, traj_y


def _simulate_walker_job(
  index: int,
  label: str,
  p_q: float,
  p_xi: float,
  max_steps: int,
  seed: int,
  x0: int,
  y0: int,
  grid_size: int,
) -> dict:
  """Worker: simulate one walker (must be top-level for ProcessPoolExecutor)."""
  coin = make_general_coin(p_q, p_xi)[0]
  traj_x, traj_y = simulate_single_trajectory(
    coin,
    max_steps,
    x0=x0,
    y0=y0,
    seed=seed,
    grid_size=grid_size,
  )
  return {
    "index": index,
    "label": label,
    "p_q": p_q,
    "p_xi": p_xi,
    "seed": seed,
    "traj_x": traj_x,
    "traj_y": traj_y,
    "final_position": [int(traj_x[-1]), int(traj_y[-1])],
  }


def trajectory_segment_colors(
  color: str,
  n_segments: int,
  *,
  pale_fraction: float = COLOR_PALE_FRACTION,
) -> np.ndarray:
  """Return solid RGB colors from a pale tint to the full walker color."""
  base = np.array(to_rgb(color))
  start = 1.0 - pale_fraction * (1.0 - base)
  t = np.linspace(0.0, 1.0, n_segments)[:, None]
  return start + (base - start) * t


def plot_trajectory_time_gradient(
  ax: plt.Axes,
  traj_x: np.ndarray,
  traj_y: np.ndarray,
  color: str,
  *,
  linewidth: float = 0.8,
  pale_fraction: float = COLOR_PALE_FRACTION,
  grid_size: int = GRID_SIZE,
) -> None:
  """Plot a trajectory with solid colors that intensify along time.

  Segments that wrap across periodic boundaries are omitted so the plot
  does not draw long lines across the domain.
  """
  points = np.column_stack([traj_x, traj_y]).astype(float)
  segments = np.stack([points[:-1], points[1:]], axis=1)
  wrap_jump = grid_size / 2.0
  keep = (
    (np.abs(segments[:, 1, 0] - segments[:, 0, 0]) <= wrap_jump)
    & (np.abs(segments[:, 1, 1] - segments[:, 0, 1]) <= wrap_jump)
  )
  segments = segments[keep]
  colors = trajectory_segment_colors(color, len(segments), pale_fraction=pale_fraction)
  lc = LineCollection(segments, colors=colors, linewidths=linewidth, capstyle="round")
  ax.add_collection(lc)
  ax.scatter(
    traj_x[0],
    traj_y[0],
    facecolors="white",
    edgecolors="black",
    s=48,
    linewidths=1.5,
    zorder=6,
  )
  ax.scatter(
    traj_x[-1],
    traj_y[-1],
    marker="s",
    facecolors="white",
    edgecolors="black",
    s=48,
    linewidths=1.5,
    zorder=6,
  )


def main() -> None:
  x0 = GRID_SIZE // 2
  y0 = GRID_SIZE // 2

  jobs = [
    (i, label, p_q, p_xi, T, SEED + i, x0, y0, GRID_SIZE)
    for i, (label, p_q, p_xi) in enumerate(WALKERS)
  ]
  with ProcessPoolExecutor(max_workers=len(WALKERS)) as executor:
    results = list(executor.map(_simulate_walker_job, *zip(*jobs)))

  results = sorted(results, key=lambda r: r["index"])

  fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
  meta_walkers = []

  for result, color, ax in zip(results, COLORS, axes.ravel()):
    traj_x = result["traj_x"]
    traj_y = result["traj_y"]
    plot_trajectory_time_gradient(
      ax, traj_x, traj_y, color, linewidth=3.5, grid_size=GRID_SIZE
    )
    ax.set_title(result["label"], fontsize=26, pad=12)
    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(-0.5, GRID_SIZE - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    meta_walkers.append({
      "label": result["label"],
      "p_q": result["p_q"],
      "p_xi": result["p_xi"],
      "seed": result["seed"],
      "start": [x0, y0],
      "final_position": result["final_position"],
    })

  fig.subplots_adjust(wspace=0.28)
  SAVE_DIR.mkdir(parents=True, exist_ok=True)
  fig.savefig(OUTPUT_PDF, format="pdf", dpi=500, bbox_inches="tight")
  fig.savefig(OUTPUT_SVG, format="svg", dpi=500, bbox_inches="tight")
  plt.close(fig)

  meta = {
    "figure": "figure1_trajectories",
    "T": T,
    "grid_size": GRID_SIZE,
    "boundary": "periodic",
    "start": [x0, y0],
    "base_seed": int(SEED),
    "color_pale_fraction": COLOR_PALE_FRACTION,
    "time_encoding": "solid_color_gradient",
    "layout": "1x4",
    "walkers": meta_walkers,
    "output_pdf": str(OUTPUT_PDF),
    "output_svg": str(OUTPUT_SVG),
  }
  with open(OUTPUT_META, "w") as f:
    json.dump(meta, f, indent=2)

  print(f"Saved figure to {OUTPUT_PDF}")
  print(f"Saved figure to {OUTPUT_SVG}")
  print(f"Saved metadata to {OUTPUT_META}")


if __name__ == "__main__":
  main()
