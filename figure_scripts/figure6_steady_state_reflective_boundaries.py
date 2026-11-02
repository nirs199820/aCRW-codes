"""Figure 6: steady-state comparison for reflective boundary conditions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT
sys.path.insert(0, str(ROOT))

from src_python_new import (
  evolve_distribution_4x4,
  make_acp_coin_4x4,
  point_initial_state,
  snapshot_to_rho_array,
)

CMAP_NAME = "plotly_hot"
CMAP = LinearSegmentedColormap.from_list(
  CMAP_NAME,
  [
    (0.0, (0.0, 0.0, 0.0)),
    (0.3, (230 / 255, 0.0, 0.0)),
    (0.6, (1.0, 210 / 255, 0.0)),
    (1.0, (1.0, 1.0, 1.0)),
  ],
)

P_Q = 0.6
P_XI = 0.2
N = 41
T_TRANSIENT = 150
T = 5_000
START_ORIENTATION = "uniform"
RNG_SEED = 42
BOUNDARY = "reflecting"
ORIGIN = (N // 2, N // 2)

BOUNDARY_MODES = [
  ("Reversing", "reflective_infinite_force"),
  ("Randomized", "randomized"),
  ("Coin", "reflecting"),
]
# Palette matched to figure1/3 (red / blue / green)
PROFILE_COLORS = {
  "Reversing": "#D21517",
  "Randomized": "#3F70AD",
  "Coin": "#32A769",
}

SAVE_DIR = REPO_ROOT / "Figures -ACRW/Fig6 - Steady State Boundary Conditions"
OUTPUT_PDF = SAVE_DIR / "figure6_steady_state_reflective_boundaries.pdf"
OUTPUT_SVG = SAVE_DIR / "figure6_steady_state_reflective_boundaries.svg"
OUTPUT_META = SAVE_DIR / "figure6_steady_state_reflective_boundaries_meta.json"

_TIMES_PATH = Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf")
if _TIMES_PATH.exists():
  font_manager.fontManager.addfont(str(_TIMES_PATH))

plt.rcParams.update({
  "font.family": "serif",
  "font.serif": ["Times New Roman", "Times", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
  "mathtext.fontset": "stix",
  "mathtext.default": "regular",
  "axes.labelsize": 22,
  "axes.titlesize": 18,
  "xtick.labelsize": 20,
  "ytick.labelsize": 20,
  "legend.fontsize": 16,
})


def _run_density_snapshots(
  coin: np.ndarray,
  reflection_coin_type: str,
) -> tuple[np.ndarray, np.ndarray]:
  """Return (rho_transient, rho_steady) at T_TRANSIENT and T."""
  rng = np.random.default_rng(RNG_SEED)
  grid = point_initial_state(N, ORIGIN, orientation=START_ORIENTATION, rng=rng)
  save_times = np.array([T_TRANSIENT, T], dtype=int)
  _, snapshots = evolve_distribution_4x4(
    grid,
    coin,
    T,
    save_times,
    BOUNDARY,
    reflection_coin_type,
  )
  rho_transient = snapshot_to_rho_array(snapshots[T_TRANSIENT], N)
  rho_steady = snapshot_to_rho_array(snapshots[T], N)
  return rho_transient, rho_steady


def _plot_heatmap_row(
  fig: plt.Figure,
  gs,
  row: int,
  rho_by_mode: list[tuple[str, np.ndarray]],
  *,
  show_titles: bool,
  row_label: str,
) -> None:
  vmax = max(float(rho.max()) for _, rho in rho_by_mode)
  for idx, (label, rho) in enumerate(rho_by_mode):
    ax = fig.add_subplot(gs[row, idx])
    im = ax.imshow(
      rho.T / vmax,
      origin="lower",
      cmap=CMAP,
      vmin=0.0,
      vmax=1.0,
    )
    ax.scatter(ORIGIN[0], ORIGIN[1], c="cyan", s=12, marker="x")
    if show_titles:
      ax.set_title(label)
    ax.set_xticks([], labels=[])
    ax.set_yticks([], labels=[])
    if idx == 0:
      ax.set_ylabel(row_label, fontsize=16)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(r"$\rho/\rho_{\rm max}$")
    cbar.ax.tick_params(labelsize=16)


def _make_figure(
  transient_by_mode: list[tuple[str, np.ndarray]],
  steady_by_mode: list[tuple[str, str, np.ndarray]],
) -> plt.Figure:
  fig = plt.figure(figsize=(13, 11), layout="constrained")
  gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.8])

  _plot_heatmap_row(
    fig,
    gs,
    0,
    [(label, rho) for label, rho in transient_by_mode],
    show_titles=True,
    row_label=f"t = {T_TRANSIENT}",
  )
  _plot_heatmap_row(
    fig,
    gs,
    1,
    [(label, rho) for label, _, rho in steady_by_mode],
    show_titles=False,
    row_label=f"t = {T}",
  )

  steady_vmax = max(float(rho.max()) for _, _, rho in steady_by_mode)
  x_coords = np.arange(N)
  ax_profile = fig.add_subplot(gs[2, :])
  for label, _, rho in steady_by_mode:
    profile = rho.mean(axis=1)
    ax_profile.plot(
      x_coords,
      profile / steady_vmax,
      lw=2.5,
      color=PROFILE_COLORS[label],
      label=label,
    )

  ax_profile.set_xlabel("x")
  ax_profile.set_ylabel(r"$\bar{\rho}(x)$")
  ax_profile.set_xlim(0, N - 1)
  ax_profile.set_xticks(
    np.arange(0, N, N // 4),
    labels=["0", f"{N // 4}", f"{N // 2}", f"{3 * N // 4}", f"{N - 1}"],
  )
  y_ticks = ax_profile.get_yticks()[0:-1:2]
  ax_profile.set_yticks(y_ticks, labels=[f"{y:.2f}" for y in y_ticks])
  ax_profile.legend(frameon=False)

  return fig


def main() -> None:
  coin = make_acp_coin_4x4(P_Q, P_XI)
  print(f"Coin row sums: {coin.sum(axis=1)}")
  print(
    f"Running comparison on N={N}, T_transient={T_TRANSIENT}, "
    f"T_steady={T}, origin={ORIGIN}"
  )

  transient_by_mode: list[tuple[str, np.ndarray]] = []
  steady_by_mode: list[tuple[str, str, np.ndarray]] = []
  for label, reflection_coin_type in BOUNDARY_MODES:
    print(f"Simulating {label} walls ({reflection_coin_type}) ...")
    rho_transient, rho_steady = _run_density_snapshots(coin, reflection_coin_type)
    transient_by_mode.append((label, rho_transient))
    steady_by_mode.append((label, reflection_coin_type, rho_steady))
    print(f"  transient total probability: {rho_transient.sum():.6f}")
    print(f"  transient max rho: {rho_transient.max():.6e}")
    print(f"  steady total probability: {rho_steady.sum():.6f}")
    print(f"  steady max rho: {rho_steady.max():.6e}")

  fig = _make_figure(transient_by_mode, steady_by_mode)

  SAVE_DIR.mkdir(parents=True, exist_ok=True)
  fig.savefig(OUTPUT_PDF, format="pdf", dpi=300, bbox_inches="tight")
  fig.savefig(OUTPUT_SVG, format="svg", dpi=300, bbox_inches="tight")
  plt.close(fig)

  meta = {
    "figure": "figure6_steady_state_reflective_boundaries",
    "p_q": P_Q,
    "p_xi": P_XI,
    "N": N,
    "T_transient": T_TRANSIENT,
    "T": T,
    "start_orientation": START_ORIENTATION,
    "rng_seed": RNG_SEED,
    "boundary": BOUNDARY,
    "origin": list(ORIGIN),
    "profile": "y_averaged_rho_bar_x_steady",
    "cmap": CMAP_NAME,
    "modes": [
      {
        "label": label,
        "reflection_coin_type": reflection_coin_type,
        "transient_total_probability": float(rho_t.sum()),
        "transient_rho_max": float(rho_t.max()),
        "final_total_probability": float(rho_s.sum()),
        "rho_max": float(rho_s.max()),
      }
      for (label, rho_t), (_, reflection_coin_type, rho_s) in zip(
        transient_by_mode, steady_by_mode
      )
    ],
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
