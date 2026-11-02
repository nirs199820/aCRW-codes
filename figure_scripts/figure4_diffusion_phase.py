"""Figure 4: 2D heatmap of normalized GK diffusion coefficient over (q_p, q_xi)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.ticker import FixedLocator, LogFormatterMathtext, ScalarFormatter

ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT
sys.path.insert(0, str(ROOT))

SAVE_DIR = REPO_ROOT / "Figures -ACRW/Fig4 - Diffusion Phase Diagram"
OUTPUT_PDF = SAVE_DIR / "figure4_diffusion_phase.pdf"
OUTPUT_SVG = SAVE_DIR / "figure4_diffusion_phase.svg"
OUTPUT_META = SAVE_DIR / "figure4_diffusion_phase_meta.json"

GRID_RESOLUTION = 1000
COLORBAR_SCALE = "linear"  # "linear" or "log"
COLORBAR_LINEAR_VMAX = 10.0
COLORBAR_LOG_VMIN = 0.01
COLORBAR_LOG_VMAX = 100.0
DENOMINATOR_EPS = 1e-8

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


def normalized_gk_diffusion(q_p: np.ndarray, q_xi: np.ndarray) -> np.ndarray:
  """Return D_GK / D_0 from Eq. (Deff-GK), with D_0 = ell^2 / (4 Delta t)."""
  numerator = 1.0 - q_p ** 2 - q_xi ** 2
  denominator = (1.0 - q_p) ** 2 + q_xi ** 2
  with np.errstate(divide="ignore", invalid="ignore"):
    ratio = numerator / denominator
  ratio = np.asarray(ratio, dtype=np.float64)
  ratio[~np.isfinite(ratio)] = np.nan
  return ratio


def unit_disk_mask(q_p: np.ndarray, q_xi: np.ndarray) -> np.ndarray:
  """Unit disk: q_p^2 + q_xi^2 <= 1."""
  return q_p ** 2 + q_xi ** 2 <= 1.0 + 1e-12


def physical_diamond_mask(q_p: np.ndarray, q_xi: np.ndarray) -> np.ndarray:
  """Physically accessible region: |q_p| + |q_xi| <= 1."""
  return np.abs(q_p) + np.abs(q_xi) <= 1.0 + 1e-12


def diamond_boundary() -> tuple[np.ndarray, np.ndarray]:
  """Vertices of the |q_p| + |q_xi| = 1 diamond."""
  q_p = np.array([1.0, 0.0, -1.0, 0.0, 1.0])
  q_xi = np.array([0.0, 1.0, 0.0, -1.0, 0.0])
  return q_p, q_xi


def enhancement_boundary_circle(n_points: int = 400) -> tuple[np.ndarray, np.ndarray]:
  """Circle (q_p - 1/2)^2 + q_xi^2 = 1/4: boundary between enhanced and reduced D."""
  theta = np.linspace(0.0, 2.0 * np.pi, n_points)
  q_p = 0.5 + 0.5 * np.cos(theta)
  q_xi = 0.5 * np.sin(theta)
  return q_p, q_xi


def colorbar_config(scale: str) -> dict:
  """Return clip bounds, norm, and tick style for the selected colorbar scale."""
  if scale == "linear":
    return {
      "clip_min": 0.0,
      "clip_max": COLORBAR_LINEAR_VMAX,
      "norm": mcolors.TwoSlopeNorm(
        vmin=0.0,
        vcenter=1.0,
        vmax=COLORBAR_LINEAR_VMAX,
      ),
      "use_log_ticks": False,
      "scale_label": "linear",
    }
  if scale == "log":
    return {
      "clip_min": COLORBAR_LOG_VMIN,
      "clip_max": COLORBAR_LOG_VMAX,
      "norm": mcolors.LogNorm(vmin=COLORBAR_LOG_VMIN, vmax=COLORBAR_LOG_VMAX),
      "use_log_ticks": True,
      "scale_label": "log",
    }
  raise ValueError(f"COLORBAR_SCALE must be 'linear' or 'log', got {scale!r}")


def colorbar_ticks(scale: str, clip_min: float, clip_max: float) -> list[float]:
  """Build colorbar ticks, always including 1 when it lies in range."""
  reference = 1.0
  if scale == "log":
    exp_min = int(np.floor(np.log10(clip_min)))
    exp_max = int(np.ceil(np.log10(clip_max)))
    ticks = {10.0 ** exp for exp in range(exp_min, exp_max + 1)}
  else:
    candidates = np.concatenate([np.linspace(clip_min, 1.0, 3), np.floor(np.linspace(1.0, clip_max, 3)[1:])])
    ticks = {float(np.round(t, 3)) for t in candidates}
    ticks.update({clip_min, clip_max})

  if clip_min <= reference <= clip_max:
    ticks.add(reference)

  return sorted(t for t in ticks if clip_min - 1e-12 <= t <= clip_max + 1e-12)


def apply_colorbar_ticks(cbar, scale: str, clip_min: float, clip_max: float) -> None:
  ticks = colorbar_ticks(scale, clip_min, clip_max)
  cbar.ax.yaxis.set_major_locator(FixedLocator(ticks))
  if scale == "log":
    cbar.ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
  else:
    cbar.ax.yaxis.set_major_formatter(ScalarFormatter())


def main() -> None:
  q_p = np.linspace(-1.0, 1.0, GRID_RESOLUTION)
  q_xi = np.linspace(-1.0, 1.0, GRID_RESOLUTION)
  qp_grid, qxi_grid = np.meshgrid(q_p, q_xi, indexing="xy")

  ratio = normalized_gk_diffusion(qp_grid, qxi_grid)
  denominator = (1.0 - qp_grid) ** 2 + qxi_grid ** 2
  unit_disk = unit_disk_mask(qp_grid, qxi_grid)
  diamond = physical_diamond_mask(qp_grid, qxi_grid)
  ratio_ok = unit_disk & (denominator > DENOMINATOR_EPS) & (ratio >= 0.0)
  cbar_cfg = colorbar_config(COLORBAR_SCALE)
  ratio_display = np.clip(ratio, cbar_cfg["clip_min"], cbar_cfg["clip_max"])

  gray_layer = np.where(ratio_ok & ~diamond, ratio_display, np.nan)
  color_layer = np.where(ratio_ok & diamond, ratio_display, np.nan)

  norm = cbar_cfg["norm"]
  extent = (-1.0, 1.0, -1.0, 1.0)

  fig, ax = plt.subplots(figsize=(7, 6))
  ax.imshow(
    gray_layer,
    origin="lower",
    extent=extent,
    cmap="RdBu_r",
    norm=norm,
    interpolation="bilinear",
    aspect="equal",
    alpha=0.5,
    zorder=1,
  )
  im = ax.imshow(
    color_layer,
    origin="lower",
    extent=extent,
    cmap="RdBu_r",
    norm=norm,
    interpolation="bilinear",
    aspect="equal",
    zorder=2,
  )

  circle_qp, circle_qxi = enhancement_boundary_circle()
  ax.plot(
    circle_qp,
    circle_qxi,
    color="black",
    lw=1.0,
    ls="--",
    zorder=3,
  )

  diamond_qp, diamond_qxi = diamond_boundary()
  ax.plot(diamond_qp, diamond_qxi, color="black", lw=2.0, zorder=4)

  cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
  cbar.set_label(r"$D_{\mathrm{eff}} / D_0$")
  apply_colorbar_ticks(
    cbar, COLORBAR_SCALE, cbar_cfg["clip_min"], cbar_cfg["clip_max"]
  )
  cbar.ax.tick_params(labelsize=20)

  ax.set_xlabel(r"$q_p$")
  ax.set_ylabel(r"$q_c$")
  ax.set_xlim(-1.0, 1.0)
  ax.set_ylim(-1.0, 1.0)
  ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0], labels=["-1", "-0.5", "0", "0.5", "1"])
  ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0], labels=["-1", "-0.5", "0", "0.5", "1"])
  fig.tight_layout()

  SAVE_DIR.mkdir(parents=True, exist_ok=True)
  fig.savefig(OUTPUT_PDF, format="pdf", dpi=300, bbox_inches="tight")
  fig.savefig(OUTPUT_SVG, format="svg", dpi=300, bbox_inches="tight")
  plt.close(fig)

  meta = {
    "figure": "figure4_diffusion_phase",
    "formula": "D_GK/D_0 = (1 - q_p^2 - q_xi^2) / ((1 - q_p)^2 + q_xi^2)",
    "enhancement_boundary": "(q_p - 1/2)^2 + q_xi^2 = 1/4",
    "physical_region": "|q_p| + |q_xi| <= 1",
    "display_region": "q_p^2 + q_xi^2 <= 1",
    "outside_diamond_style": "grayscale",
    "grid_resolution": GRID_RESOLUTION,
    "colorbar_scale": COLORBAR_SCALE,
    "colorbar_linear_vmax": COLORBAR_LINEAR_VMAX,
    "colorbar_log_vmin": COLORBAR_LOG_VMIN,
    "colorbar_log_vmax": COLORBAR_LOG_VMAX,
    "colormap": "RdBu_r",
    "denominator_eps": DENOMINATOR_EPS,
    "diamond_max_ratio_finite": float(np.nanmax(ratio[np.where(ratio_ok & diamond)])),
    "output_pdf": str(OUTPUT_PDF),
    "output_svg": str(OUTPUT_SVG),
  }
  with open(OUTPUT_META, "w") as f:
    json.dump(meta, f, indent=2)

  print(f"D_GK/D_0 max (finite, in diamond): {meta['diamond_max_ratio_finite']:.4f}")
  if COLORBAR_SCALE == "linear":
    print(
      f"Colorbar range (linear): [0, 1 (white), {COLORBAR_LINEAR_VMAX}]"
    )
  else:
    print(
      f"Colorbar range (log): [{COLORBAR_LOG_VMIN}, 1 (white), {COLORBAR_LOG_VMAX}]"
    )
  print(f"Saved figure to {OUTPUT_PDF}")
  print(f"Saved figure to {OUTPUT_SVG}")
  print(f"Saved metadata to {OUTPUT_META}")


if __name__ == "__main__":
  main()
