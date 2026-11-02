"""Figure 3: ensemble MSD, velocity autocorrelation, and cross-correlation with theory."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT
sys.path.insert(0, str(ROOT))

from src_python_new import (
  calculate_direction_crosscorr_batched_at_lags_numba,
  calculate_ensemble_msd_from_origin_at_lags_numba,
  calculate_velocity_autocorr_batched_at_lags_numba,
  fit_msd_power_law,
  make_general_coin,
  msd_lag_schedule,
  simulate_free_diffusion_ensemble_fixed_single_coin_batched,
)

N = 100_000
T = 501
DT_save = 1
SEED = 42
N_PER_DECADE = 20
LINEAR_SHORT_MAX = 50
T_FIT_MIN = 100
T_FIT_MAX = int(T)

SAVE_DIR = REPO_ROOT / "Figures -ACRW/Fig3 - MSD and Theory"
OUTPUT_MSD_PDF = SAVE_DIR / "figure3_msd.pdf"
OUTPUT_MSD_SVG = SAVE_DIR / "figure3_msd.svg"
OUTPUT_VACF_PDF = SAVE_DIR / "figure3_VACF.pdf"
OUTPUT_VACF_SVG = SAVE_DIR / "figure3_VACF.svg"
OUTPUT_VCCF_PDF = SAVE_DIR / "figure3_VCCF.pdf"
OUTPUT_VCCF_SVG = SAVE_DIR / "figure3_VCCF.svg"
OUTPUT_META = SAVE_DIR / "figure3_msd_meta.json"

WALKERS = [
  ("Random", 0.0, 0.0),
  ("Chiral", 0.0, 0.5),
  ("Active Chiral", 0.5, 0.2),
  ("Anti-persistent", -0.5, 0.2),
]
# Palette matched to figure1 (red / blue / green / purple)
COLORS = ["#D21517", "#3F70AD", "#32A769", "#972B91"]

_TIMES_PATH = Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf")
if _TIMES_PATH.exists():
  font_manager.fontManager.addfont(str(_TIMES_PATH))

plt.rcParams.update({
  "font.family": "serif",
  "font.serif": ["Times New Roman", "Times", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
  "mathtext.fontset": "stix",
  "mathtext.default": "regular",
  "axes.labelsize": 22,
  "xtick.labelsize": 20,
  "ytick.labelsize": 20,
  "legend.fontsize": 18,
})


def expected_diffusion_constant(p_q: float, p_xi: float) -> float:
  if p_q == 1.0:
    return 1.0
  return ((1.0 - p_q ** 2 - p_xi ** 2) / (((1.0 - p_q) ** 2 + p_xi ** 2))) / 4.0


def persistence_eigenvalue_params(p_q: float, p_xi: float) -> tuple[float, float, float]:
  """Return rho, Omega, ln(rho) for lambda^n = rho^n exp(i Omega n)."""
  rho = float(np.hypot(p_q, p_xi))
  if p_q == 0.0:
    if p_xi == 0.0:
      omega = 0.0
    elif p_xi > 0.0:
      omega = float(np.pi / 2.0)
    else:
      omega = float(-np.pi / 2.0)
  else:
    omega = float(np.arctan2(p_xi, p_q))
  ln_rho = float(np.log(rho)) if rho > 0.0 else float("-inf")
  return rho, omega, ln_rho


def theoretical_vacf(lags: np.ndarray, p_q: float, p_xi: float) -> np.ndarray:
  """Theory for avg velocity autocorrelation: Re(lambda^n)/2 = rho^n cos(Omega n)/2."""
  c_parallel, _ = theoretical_velocity_crosscorr(lags, p_q, p_xi)
  return c_parallel / 2.0


def theoretical_velocity_crosscorr(
  lags: np.ndarray, p_q: float, p_xi: float
) -> tuple[np.ndarray, np.ndarray]:
  """Theory for velocity cross-correlations from lambda^n = rho^n exp(i Omega n)."""
  lags = np.asarray(lags, dtype=np.float64)
  if p_q == 0.0 and p_xi == 0.0:
    zeros = np.zeros_like(lags)
    return zeros, zeros
  rho, omega, _ = persistence_eigenvalue_params(p_q, p_xi)
  envelope = rho ** lags
  c_parallel = envelope * np.cos(omega * lags)
  c_perp = envelope * np.sin(omega * lags)
  return c_parallel, c_perp


def _simulate_and_observables_job(
  index: int,
  label: str,
  p_q: float,
  p_xi: float,
  n_particles: int,
  max_steps: int,
  dt_save: int,
  seed: int,
) -> dict:
  """Worker: simulate one ensemble and compute MSD + velocity correlations."""
  _ = seed
  coin = make_general_coin(p_q, p_xi)[0]
  save_times = np.arange(0, max_steps, dt_save).astype(int)
  traj_x, traj_y, dir_x, dir_y = simulate_free_diffusion_ensemble_fixed_single_coin_batched(
    coin,
    save_times,
    N_particles=n_particles,
    x0=0,
    y0=0,
    max_steps=max_steps,
    return_directions=True,
  )

  msd_lags = msd_lag_schedule(
    max_steps,
    n_per_decade=N_PER_DECADE,
    linear_short_max=LINEAR_SHORT_MAX,
    lag_max=max_steps - 1,
  )
  msd = calculate_ensemble_msd_from_origin_at_lags_numba(traj_x, traj_y, msd_lags)
  msd_times = msd_lags * dt_save
  coeffs, fit_mask = fit_msd_power_law(msd_times, msd, T_FIT_MIN, T_FIT_MAX)
  d_active = float(np.exp(coeffs[1]) / 4.0) if np.isfinite(coeffs[1]) else float("nan")
  d_expected = expected_diffusion_constant(p_q, p_xi)

  vacf_lags = np.arange(1, max_steps, dtype=np.int64)
  dx_corr, dy_corr = calculate_velocity_autocorr_batched_at_lags_numba(
    dir_x, dir_y, vacf_lags
  )
  vacf_times = vacf_lags * dt_save
  vacf = (dx_corr + dy_corr) / 2.0
  c_parallel, c_perp = calculate_direction_crosscorr_batched_at_lags_numba(
    dir_x, dir_y, vacf_lags
  )
  rho_theory, omega_theory, ln_rho_theory = persistence_eigenvalue_params(p_q, p_xi)

  return {
    "index": index,
    "label": label,
    "p_q": p_q,
    "p_xi": p_xi,
    "seed": seed,
    "times": msd_times,
    "msd": msd,
    "coeffs": coeffs,
    "fit_mask": fit_mask,
    "D_active": d_active,
    "expected_D": d_expected,
    "vacf_lags": vacf_lags,
    "vacf_times": vacf_times,
    "vacf": vacf,
    "c_parallel": c_parallel,
    "c_perp": c_perp,
    "rho_theory": rho_theory,
    "omega_theory": omega_theory,
    "ln_rho_theory": ln_rho_theory,
  }


def main() -> None:
  jobs = [
    (i, label, p_q, p_xi, N, T, DT_save, SEED + i)
    for i, (label, p_q, p_xi) in enumerate(WALKERS)
  ]
  with ProcessPoolExecutor(max_workers=len(WALKERS)) as executor:
    results = list(executor.map(_simulate_and_observables_job, *zip(*jobs)))

  results = sorted(results, key=lambda r: r["index"])

  fig_msd, ax_msd = plt.subplots(figsize=(8, 5.2))
  fig_vacf, ax_vacf = plt.subplots(figsize=(6, 4))
  fig_vccf, (ax_parallel, ax_perp) = plt.subplots(2, 1, figsize=(4, 4))
  meta_walkers = []
  lw_data = 1.8
  lw_theory = 3.2
  z_data = 3
  z_theory = 1

  for result, color in zip(results, COLORS):
    times = result["times"]
    msd = result["msd"]
    coeffs = result["coeffs"]
    label = result["label"]
    vacf_times = result["vacf_times"]
    vacf = result["vacf"]
    vacf_lags = result["vacf_lags"]
    c_parallel = result["c_parallel"]
    c_perp = result["c_perp"]
    theory_parallel, theory_perp = theoretical_velocity_crosscorr(
      vacf_lags, result["p_q"], result["p_xi"]
    )

    ax_msd.loglog(
      times,
      4.0 * result["expected_D"] * times,
      "--",
      color=color,
      lw=lw_theory,
      zorder=z_theory,
    )
    ax_msd.loglog(
      times, msd, "-", color=color, lw=lw_data, label=label, zorder=z_data
    )

    ax_vacf.plot(
      vacf_times,
      theoretical_vacf(vacf_lags, result["p_q"], result["p_xi"]),
      "--",
      color=color,
      lw=lw_theory,
      zorder=z_theory,
    )
    ax_vacf.plot(
      vacf_times, vacf, "-", color=color, lw=lw_data, label=label, zorder=z_data
    )

    ax_parallel.plot(
      vacf_times, theory_parallel, "--", color=color, lw=lw_theory, zorder=z_theory
    )
    ax_parallel.plot(
      vacf_times, c_parallel, "-", color=color, lw=lw_data, label=label, zorder=z_data
    )
    ax_perp.plot(
      vacf_times, theory_perp, "--", color=color, lw=lw_theory, zorder=z_theory
    )
    ax_perp.plot(
      vacf_times, c_perp, "-", color=color, lw=lw_data, label=label, zorder=z_data
    )

    print(
      f"{label}: m={coeffs[0]:.3f}, "
      f"D_active={result['D_active']:.4f}, expected_D={result['expected_D']:.4f}, "
      f"rho={result['rho_theory']:.4f}, Omega={result['omega_theory']:.4f}, "
      f"ln(rho)={result['ln_rho_theory']:.4f}"
    )
    meta_walkers.append({
      "label": label,
      "p_q": result["p_q"],
      "p_xi": result["p_xi"],
      "seed": result["seed"],
      "m": float(coeffs[0]),
      "b": float(coeffs[1]),
      "D_active": result["D_active"],
      "expected_D": result["expected_D"],
      "rho": result["rho_theory"],
      "Omega": result["omega_theory"],
      "ln_rho": result["ln_rho_theory"],
    })

  ax_msd.set_xlabel("t")
  ax_msd.set_ylabel("MSD")
  ax_msd.set_xticks([1, 10, 100], labels=["1", "10", "100"])
  ax_msd.set_yticks([1, 10, 100], labels=["1", "10", "100"])
  y_min, y_max = ax_msd.get_ylim()
  ax_msd.set_ylim(y_min, y_max * 2.5)
  ax_msd.legend(
    loc="upper left",
    frameon=False,
    fontsize=13,
    handlelength=1.6,
    labelspacing=0.25,
    borderaxespad=0.3,
  )
  fig_msd.tight_layout()

  ax_vacf.set_xlabel("t")
  ax_vacf.set_ylabel(r"$\langle v_i(t)\, v_i(t+\Delta t)\rangle$")
  ax_vacf.set_xlim(1, 10)
  ax_vacf.set_xticks([1, 5, 10], labels=["1", "5", "10"])
  y_ticks_vacf = ax_vacf.get_yticks()
  ax_vacf.set_yticks(y_ticks_vacf, labels=[f"{y:.1f}" for y in y_ticks_vacf])
  ax_vacf.legend(loc="upper right", frameon=False, fontsize=13)
  fig_vacf.tight_layout()

  ax_parallel.set_xlabel("")
  ax_parallel.set_ylabel(r"$\mathit{C}_{\mathit{V}}^{\mathrm{diag}}$")
  ax_parallel.set_xticks([1, 5, 10], labels=["1", "5", "10"])
  y_ticks_parallel = [-0.4,0,0.4]
  ax_parallel.set_yticks(y_ticks_parallel, labels=[f"{y:.1f}" for y in y_ticks_parallel])
  ax_parallel.set_xlim(1, 10)
  #ax_parallel.legend(loc="upper right", frameon=False)
  ax_perp.set_xlabel("")
  ax_perp.set_ylabel(r"$\mathit{C}_{\mathit{V}}^{\mathrm{off}}$")
  ax_perp.set_xlim(1, 10)
  ax_perp.set_xticks([1, 5, 10], labels=["1", "5", "10"])
  y_ticks_perp = [-0.4,0,0.4]
  ax_perp.set_yticks(y_ticks_perp, labels=[f"{y:.1f}" for y in y_ticks_perp])
  fig_vccf.tight_layout()

  SAVE_DIR.mkdir(parents=True, exist_ok=True)
  fig_msd.savefig(OUTPUT_MSD_PDF, format="pdf", dpi=300, bbox_inches="tight")
  fig_msd.savefig(OUTPUT_MSD_SVG, format="svg", dpi=300, bbox_inches="tight")
  fig_vacf.savefig(OUTPUT_VACF_PDF, format="pdf", dpi=300, bbox_inches="tight")
  fig_vacf.savefig(OUTPUT_VACF_SVG, format="svg", dpi=300, bbox_inches="tight")
  fig_vccf.savefig(OUTPUT_VCCF_PDF, format="pdf", dpi=300, bbox_inches="tight")
  fig_vccf.savefig(OUTPUT_VCCF_SVG, format="svg", dpi=300, bbox_inches="tight")
  plt.close(fig_msd)
  plt.close(fig_vacf)
  plt.close(fig_vccf)

  meta = {
    "figure": "figure3_msd_vacf_vccf",
    "N": N,
    "T": T,
    "DT_save": DT_save,
    "base_seed": SEED,
    "t_fit_min": T_FIT_MIN,
    "t_fit_max": T_FIT_MAX,
    "n_per_decade": N_PER_DECADE,
    "linear_short_max": LINEAR_SHORT_MAX,
    "walkers": meta_walkers,
    "output_msd_pdf": str(OUTPUT_MSD_PDF),
    "output_msd_svg": str(OUTPUT_MSD_SVG),
    "output_vacf_pdf": str(OUTPUT_VACF_PDF),
    "output_vacf_svg": str(OUTPUT_VACF_SVG),
    "output_vccf_pdf": str(OUTPUT_VCCF_PDF),
    "output_vccf_svg": str(OUTPUT_VCCF_SVG),
  }
  with open(OUTPUT_META, "w") as f:
    json.dump(meta, f, indent=2)

  print(f"Saved figure to {OUTPUT_MSD_PDF}")
  print(f"Saved figure to {OUTPUT_MSD_SVG}")
  print(f"Saved figure to {OUTPUT_VACF_PDF}")
  print(f"Saved figure to {OUTPUT_VACF_SVG}")
  print(f"Saved figure to {OUTPUT_VCCF_PDF}")
  print(f"Saved figure to {OUTPUT_VCCF_SVG}")
  print(f"Saved metadata to {OUTPUT_META}")


if __name__ == "__main__":
  main()
