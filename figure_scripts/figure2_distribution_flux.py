"""Figure 2: 1x4 density + flux panels (center vs near-wall starts)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec
import numpy as np

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


_d4 = _load_module(
  "src_python_new.distribution_4x4",
  ROOT / "src_python_new" / "distribution_4x4.py",
)
make_coin_4x4 = _d4.make_coin_4x4
point_initial_state = _d4.point_initial_state
evolve_distribution_4x4 = _d4.evolve
snapshot_to_rho_array = _d4.snapshot_to_rho_array
snapshot_to_orientation_array = _d4.snapshot_to_orientation_array
flux_from_orientation = _d4.flux_from_orientation

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

# Plotly sequential.Hot (same as acp_distribution_propagation.py)
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
NORM_MODE = "log"  # "linear" or "log" after per-panel normalization to [0, 1]
LOG_NORM_MIN = 1e-3
FLUX_CMAP_NAME = "RdBu_r"  # blue-white-red (matplotlib diverging)
FLUX_CMAP = plt.get_cmap(FLUX_CMAP_NAME)

P_LEFT = 0.0
P_RIGHT = 0.8
P_PERSISTENT = 0.1
P_REVERSE = 0.0

N = 150
T = 2000
PLOT_TIMES = (100, 300, 500, 1000, 2000)#, 5000)
START_ORIENTATION = "uniform"
RNG_SEED = 42
BOUNDARY = "reflecting" # "periodic" | "open" | "reflecting" | "reflective_infinite_force"
REFLECTION_COIN_TYPE = "reflecting" # "randomized" | "reflective_infinite_force" | "reflecting" | "anti_reflecting" | "stay" | "scatter"

ORIGIN_CENTER = (N // 2, N // 2)
ORIGIN_WALL = (N // 2, 5)

SAVE_DIR = REPO_ROOT / "Figures -ACRW/Fig2 - Distribution and Flux - mixed (reflecting)"
OUTPUT_META = SAVE_DIR / "figure2_distribution_flux_meta.json"


def _normalize_field(values: np.ndarray) -> np.ndarray:
  vmax = max(float(values.max()), 1e-12)
  field = values / vmax
  if NORM_MODE == "log":
    return np.clip(field, LOG_NORM_MIN, 1.0)
  return field


def _color_norm() -> mcolors.Normalize:
  if NORM_MODE == "log":
    return mcolors.LogNorm(vmin=LOG_NORM_MIN, vmax=1.0)
  return mcolors.Normalize(vmin=0.0, vmax=1.0)


def _plot_rho(ax, rho: np.ndarray, origin: tuple[int, int], title: str) -> None:
  field = _normalize_field(rho.T)
  im = ax.imshow(field, origin="lower", cmap=CMAP, norm=_color_norm())
  ax.scatter(origin[0], origin[1], c="red", s=5, marker="x")
  ax.set_title(title)
  ax.set_xlabel("")
  ax.set_ylabel("")
  ax.set_xticks([],labels=[])
  ax.set_yticks([],labels=[])
  colorbar_label = r"$P/P_{\rm max}$"
  cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, orientation="horizontal")
  cbar.set_label(colorbar_label)
  cbar.ax.tick_params(labelsize=16)

def _plot_flux(ax, P: np.ndarray, origin: tuple[int, int], title: str) -> None:
  jx, jy, j_mag = flux_from_orientation(P)
  field = _normalize_field(j_mag.T)
  im = ax.imshow(field, origin="lower", cmap=FLUX_CMAP, norm=_color_norm())

  arrow_step = max(1, N // 10)
  xs = np.arange(N)
  ys = np.arange(N)
  X, Y = np.meshgrid(xs, ys)
  sl = (slice(None, None, arrow_step), slice(None, None, arrow_step))
  jx_q = jx.T[sl]
  jy_q = jy.T[sl]
  mag_q = np.hypot(jx_q, jy_q)
  mag_max = max(float(mag_q.max()), 1e-15)
  mask = mag_q > 0.005 * mag_max
  ux = np.zeros_like(jx_q)
  uy = np.zeros_like(jy_q)
  np.divide(jx_q, mag_q, out=ux, where=mask)
  np.divide(jy_q, mag_q, out=uy, where=mask)
  arrow_len = 0.9 * arrow_step
  ax.quiver(
    X[sl][mask],
    Y[sl][mask],
    ux[mask] * arrow_len,
    uy[mask] * arrow_len,
    color="white",
    edgecolors="black",
    linewidths=0.6,
    angles="xy",
    scale_units="xy",
    scale=1.0,
    width=0.012,
    headwidth=3.5,
    headlength=4.0,
    headaxislength=3.5,
    pivot="mid",
    zorder=5,
  )
  ax.scatter(origin[0], origin[1], c="red", s=5, marker="x")
  ax.set_title(title)
  ax.set_xlabel("")
  ax.set_ylabel("")
  ax.set_xticks([],labels=[])
  ax.set_yticks([],labels=[])
  colorbar_label = r"$|J|/|J|_{\rm max}$"
  cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, orientation="horizontal")
  cbar.set_label(colorbar_label)
  cbar.ax.tick_params(labelsize=16)


def _run_evolution(origin: tuple[int, int], coin: np.ndarray) -> dict[int, object]:
  rng = np.random.default_rng(RNG_SEED)
  grid = point_initial_state(N, origin, orientation=START_ORIENTATION, rng=rng)
  save_times = np.unique(np.array([0, *PLOT_TIMES, T], dtype=int))
  _, snapshots = evolve_distribution_4x4(
    grid,
    coin,
    T,
    save_times,
    BOUNDARY,
    REFLECTION_COIN_TYPE,
  )
  return snapshots


def _make_figure(
  t: int,
  snaps_center: dict[int, object],
  snaps_wall: dict[int, object],
) -> plt.Figure:
  # 1x4 line: [rho_c | rho_w]  --larger gap--  [flux_c | flux_w]
  # Spacer column (index 2) creates the larger gap between pairs.
  fig = plt.figure(figsize=(16, 4.8))
  gs = GridSpec(
    1,
    5,
    figure=fig,
    width_ratios=[1.0, 1.0, 0.28, 1.0, 1.0],
    wspace=0.08,
  )
  ax_rho_c = fig.add_subplot(gs[0, 0])
  ax_rho_w = fig.add_subplot(gs[0, 1])
  ax_flux_c = fig.add_subplot(gs[0, 3])
  ax_flux_w = fig.add_subplot(gs[0, 4])

  rho_c = snapshot_to_rho_array(snaps_center[t], N)
  rho_w = snapshot_to_rho_array(snaps_wall[t], N)
  P_c = snapshot_to_orientation_array(snaps_center[t], N)
  P_w = snapshot_to_orientation_array(snaps_wall[t], N)

  _plot_rho(ax_rho_c, rho_c, ORIGIN_CENTER, "")
  _plot_rho(ax_rho_w, rho_w, ORIGIN_WALL, "")
  _plot_flux(ax_flux_c, P_c, ORIGIN_CENTER, "")
  _plot_flux(ax_flux_w, P_w, ORIGIN_WALL, "")

  fig.suptitle("")
  return fig


def main() -> None:
  coin = make_coin_4x4(P_PERSISTENT, P_RIGHT, P_REVERSE, P_LEFT)
  print(f"Coin matrix row sums: {coin.sum(axis=1)}")
  print(f"Evolving center start {ORIGIN_CENTER} ...")
  snaps_center = _run_evolution(ORIGIN_CENTER, coin)
  print(f"Evolving wall start {ORIGIN_WALL} ...")
  snaps_wall = _run_evolution(ORIGIN_WALL, coin)

  SAVE_DIR.mkdir(parents=True, exist_ok=True)
  output_files: list[dict[str, str]] = []

  for t in PLOT_TIMES:
    fig = _make_figure(t, snaps_center, snaps_wall)
    pdf_path = SAVE_DIR / f"figure2_distribution_flux_t{t}.pdf"
    svg_path = SAVE_DIR / f"figure2_distribution_flux_t{t}.svg"
    fig.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", dpi=300, bbox_inches="tight")
    plt.close(fig)
    output_files.append({
      "t": t,
      "pdf": str(pdf_path),
      "svg": str(svg_path),
    })
    print(f"Saved t={t}: {pdf_path.name}, {svg_path.name}")

  meta = {
    "figure": "figure2_distribution_flux",
    "p_left": P_LEFT,
    "p_right": P_RIGHT,
    "p_persistent": P_PERSISTENT,
    "p_reverse": P_REVERSE,
    "N": N,
    "T": T,
    "plot_times": list(PLOT_TIMES),
    "cmap": CMAP_NAME,
    "flux_cmap": FLUX_CMAP_NAME,
    "norm_mode": NORM_MODE,
    "log_norm_min": LOG_NORM_MIN,
    "start_orientation": START_ORIENTATION,
    "rng_seed": RNG_SEED,
    "boundary": BOUNDARY,
    "reflection_coin_type": REFLECTION_COIN_TYPE,
    "layout": "1x4_densities_then_fluxes",
    "origin_center": list(ORIGIN_CENTER),
    "origin_wall": list(ORIGIN_WALL),
    "outputs": output_files,
  }
  with open(OUTPUT_META, "w") as f:
    json.dump(meta, f, indent=2)
  print(f"Saved metadata to {OUTPUT_META}")


if __name__ == "__main__":
  main()
