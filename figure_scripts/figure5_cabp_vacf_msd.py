"""Figure 5: ensemble VACF and MSD with continuum cABP theory overlays."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT
sys.path.insert(0, str(ROOT))

from src_python_new import (
  calculate_ensemble_msd_from_origin_at_lags_numba,
  calculate_velocity_autocorr_batched_at_lags_numba,
  make_general_coin,
  msd_lag_schedule,
  simulate_free_diffusion_ensemble_fixed_single_coin_batched,
)

N = 10_000
T = 1001
DT_save = 1
SEED = 42
N_PER_DECADE = 10
LINEAR_SHORT_MAX = 20
NU = 1.0
ELL = 1.0
DELTA_T = 1.0 / NU
A_EPS = 1e-12
T_MAX_PLOT = 65

SAVE_DIR = REPO_ROOT / "Figures -ACRW/Fig5 - cABP VACF and MSD"
OUTPUT_COMBINED_PDF = SAVE_DIR / "Figure5.pdf"
OUTPUT_COMBINED_SVG = SAVE_DIR / "Figure5.svg"
OUTPUT_META = SAVE_DIR / "figure5_cabp_meta.json"

WALKERS = [
  ("Strong turning", 0.6, 0.4),
  ("Weak turning", 0.9, 0.1),
]
# Palette matched to figure1/3 (blue / green)
COLORS = ["#3F70AD", "#32A769"]

_TIMES_PATH = Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf")
if _TIMES_PATH.exists():
  font_manager.fontManager.addfont(str(_TIMES_PATH))

plt.rcParams.update({
  "font.family": "serif",
  "font.serif": ["Times New Roman", "Times", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
  "mathtext.fontset": "stix",
  "mathtext.default": "regular",
  "axes.labelsize": 28,
  "xtick.labelsize": 20,
  "ytick.labelsize": 20,
  "xtick.direction": "in",
  "ytick.direction": "in",
  "legend.fontsize": 18,
})


def persistence_eigenvalue(q_p: float, q_xi: float) -> complex:
  """First-harmonic eigenvalue lambda = q_p + i q_xi."""
  return complex(q_p, q_xi)


def cabp_coefficients(
  q_p: float, q_xi: float, nu: float = NU, ell: float = ELL
) -> dict[str, float]:
  """Continuum cABP parameters from the Euler / hydrodynamic mapping."""
  d_r = nu * (1.0 - q_p)
  omega = nu * q_xi
  v0 = nu * ell
  d_tilde = 0.0
  d0_lattice = nu * ell ** 2 / 4.0
  a = (1.0 - q_p) ** 2 + q_xi ** 2
  if a < A_EPS:
    d_eff = float("inf")
    four_d_eff = float("inf")
  else:
    d_eff = d_tilde + (nu * ell ** 2 / 2.0) * (1.0 - q_p) / a
    four_d_eff = 4.0 * d_eff
  return {
    "D_r_tilde": float(d_r),
    "omega_tilde": float(omega),
    "v0_tilde": float(v0),
    "D_tilde": float(d_tilde),
    "D0_lattice": float(d0_lattice),
    "A": float(a),
    "D_eff_cabp": float(d_eff),
    "four_D_eff_cabp": float(four_d_eff),
  }


def cabp_coefficients_corr(
  q_p: float, q_xi: float, *, delta_t: float = DELTA_T, nu: float = NU, ell: float = ELL
) -> dict[str, float]:
  """Correlation-level mapping: D_r = -ln|lambda|/Delta t, omega = arg(lambda)/Delta t."""
  lam = persistence_eigenvalue(q_p, q_xi)
  abs_lam = abs(lam)
  if abs_lam <= 0.0 or abs_lam >= 1.0:
    d_r = float("inf") if abs_lam == 0.0 else 0.0
  else:
    d_r = float(-np.log(abs_lam) / delta_t)
  omega = float(np.angle(lam) / delta_t)
  v0 = nu * ell
  d_tilde = 0.0
  d0_lattice = nu * ell ** 2 / 4.0
  a = d_r ** 2 + omega ** 2
  if a < A_EPS:
    d_eff = float("inf")
    four_d_eff = float("inf")
  else:
    d_eff = d_tilde + 0.5 * v0 ** 2 * d_r / a
    four_d_eff = 4.0 * d_eff
  return {
    "D_r_corr": float(d_r),
    "omega_corr": float(omega),
    "v0_tilde": float(v0),
    "D_tilde": float(d_tilde),
    "D0_lattice": float(d0_lattice),
    "abs_lambda": float(abs_lam),
    "arg_lambda": float(np.angle(lam)),
    "D_eff_cabp_corr": float(d_eff),
    "four_D_eff_cabp_corr": float(four_d_eff),
  }


def theoretical_vacf_cabp_rates(
  t: np.ndarray, d_r: float, omega: float, v0: float
) -> np.ndarray:
  """Continuum cABP VACF: v0^2 exp(-D_r t) cos(omega t)."""
  t = np.asarray(t, dtype=np.float64)
  return v0 ** 2 * np.exp(-d_r * t) * np.cos(omega * t)


def theoretical_msd_cabp_rates(
  t: np.ndarray, d_r: float, omega: float, v0: float, d_tilde: float
) -> np.ndarray:
  """Full continuum cABP MSD from Green–Kubo for given rates."""
  t = np.asarray(t, dtype=np.float64)
  a = d_r ** 2 + omega ** 2
  if a < A_EPS:
    return 4.0 * d_tilde * t + (v0 * t) ** 2

  decay = np.exp(-d_r * t)
  cos_term = np.cos(omega * t)
  sin_term = np.sin(omega * t)
  linear = 4.0 * d_tilde * t + 2.0 * v0 ** 2 * d_r / a * t
  osc = (
    2.0
    * v0 ** 2
    / a ** 2
    * (
      (omega ** 2 - d_r ** 2) * (1.0 - decay * cos_term)
      - 2.0 * d_r * omega * decay * sin_term
    )
  )
  return linear + osc


def theoretical_vacf_cabp(
  t: np.ndarray, q_p: float, q_xi: float, nu: float = NU, ell: float = ELL
) -> np.ndarray:
  """Continuum cABP VACF with the Euler / hydrodynamic mapping."""
  coeffs = cabp_coefficients(q_p, q_xi, nu=nu, ell=ell)
  return theoretical_vacf_cabp_rates(
    t, coeffs["D_r_tilde"], coeffs["omega_tilde"], coeffs["v0_tilde"]
  )


def theoretical_msd_cabp(
  t: np.ndarray, q_p: float, q_xi: float, nu: float = NU, ell: float = ELL
) -> np.ndarray:
  """Full continuum cABP MSD with the Euler / hydrodynamic mapping."""
  coeffs = cabp_coefficients(q_p, q_xi, nu=nu, ell=ell)
  return theoretical_msd_cabp_rates(
    t,
    coeffs["D_r_tilde"],
    coeffs["omega_tilde"],
    coeffs["v0_tilde"],
    coeffs["D_tilde"],
  )


def theoretical_vacf_cabp_corr(
  t: np.ndarray, q_p: float, q_xi: float, *, delta_t: float = DELTA_T, nu: float = NU, ell: float = ELL
) -> np.ndarray:
  """Continuum cABP VACF with the correlation-level mapping."""
  coeffs = cabp_coefficients_corr(q_p, q_xi, delta_t=delta_t, nu=nu, ell=ell)
  return theoretical_vacf_cabp_rates(
    t, coeffs["D_r_corr"], coeffs["omega_corr"], coeffs["v0_tilde"]
  )


def theoretical_msd_cabp_corr(
  t: np.ndarray, q_p: float, q_xi: float, *, delta_t: float = DELTA_T, nu: float = NU, ell: float = ELL
) -> np.ndarray:
  """Full continuum cABP MSD with the correlation-level mapping."""
  coeffs = cabp_coefficients_corr(q_p, q_xi, delta_t=delta_t, nu=nu, ell=ell)
  return theoretical_msd_cabp_rates(
    t,
    coeffs["D_r_corr"],
    coeffs["omega_corr"],
    coeffs["v0_tilde"],
    coeffs["D_tilde"],
  )


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
  """Worker: simulate one ensemble and compute MSD + vector VACF."""
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

  vacf_lags = np.arange(1, max_steps, dtype=np.int64)
  dx_corr, dy_corr = calculate_velocity_autocorr_batched_at_lags_numba(
    dir_x, dir_y, vacf_lags
  )
  vacf_times = vacf_lags * dt_save
  vacf = dx_corr + dy_corr
  coeffs = cabp_coefficients(p_q, p_xi)
  coeffs_corr = cabp_coefficients_corr(p_q, p_xi)

  return {
    "index": index,
    "label": label,
    "p_q": p_q,
    "p_xi": p_xi,
    "seed": seed,
    "times": msd_times,
    "msd": msd,
    "vacf_lags": vacf_lags,
    "vacf_times": vacf_times,
    "vacf": vacf,
    **coeffs,
    **coeffs_corr,
  }


def main() -> None:
  jobs = [
    (i, label, p_q, p_xi, N, T, DT_save, SEED + i)
    for i, (label, p_q, p_xi) in enumerate(WALKERS)
  ]
  with ProcessPoolExecutor(max_workers=len(WALKERS)) as executor:
    results = list(executor.map(_simulate_and_observables_job, *zip(*jobs)))

  results = sorted(results, key=lambda r: r["index"])

  fig_combined, (ax_msd, ax_vacf) = plt.subplots(1, 2, figsize=(16.1, 5.6))
  meta_walkers = []
  lw_theory = 4.2
  lw_theory_corr = 4.0
  ms_data = 56
  z_data = 3
  z_theory = 1
  alpha = 0.9

  for result, color in zip(results, COLORS):
    times = result["times"]
    msd = result["msd"]
    label = result["label"]
    vacf_times = result["vacf_times"]
    vacf = result["vacf"]
    p_q = result["p_q"]
    p_xi = result["p_xi"]

    msd_theory = theoretical_msd_cabp(times, p_q, p_xi)
    msd_theory_corr = theoretical_msd_cabp_corr(times, p_q, p_xi)
    vacf_theory = theoretical_vacf_cabp(vacf_times, p_q, p_xi)
    vacf_theory_corr = theoretical_vacf_cabp_corr(vacf_times, p_q, p_xi)

    ax_msd.loglog(
      times, msd_theory, "--", color=color, lw=lw_theory, zorder=z_theory, alpha=alpha
    )
    ax_msd.loglog(
      times, msd_theory_corr, ":", color=color, lw=lw_theory_corr, zorder=z_theory, alpha=alpha
    )
    ax_msd.scatter(
      times,
      msd,
      s=ms_data,
      color=color,
      zorder=z_data,
      label=label,
      edgecolors="none",
      alpha=alpha,
    )

    ax_vacf.plot(
      vacf_times, vacf_theory, "--", color=color, lw=lw_theory, zorder=z_theory, alpha=alpha
    )
    ax_vacf.plot(
      vacf_times, vacf_theory_corr, ":", color=color, lw=lw_theory_corr, zorder=z_theory, alpha=alpha
    )
    vacf_mask = vacf_times <= T_MAX_PLOT
    ax_vacf.scatter(
      vacf_times[vacf_mask],
      vacf[vacf_mask],
      s=ms_data,
      color=color,
      zorder=z_data,
      label=label,
      edgecolors="none",
      alpha=alpha,
    )

    print(
      f"{label}: D_r={result['D_r_tilde']:.4f}, "
      f"omega={result['omega_tilde']:.4f}, "
      f"D_r_corr={result['D_r_corr']:.4f}, "
      f"omega_corr={result['omega_corr']:.4f}, "
      f"D_eff_cabp={result['D_eff_cabp']:.4f}, "
      f"4 D_eff_cabp={result['four_D_eff_cabp']:.4f}"
    )
    meta_walkers.append({
      "label": label,
      "q_p": p_q,
      "q_c": p_xi,
      "seed": result["seed"],
      "D_r_tilde": result["D_r_tilde"],
      "omega_tilde": result["omega_tilde"],
      "D_r_corr": result["D_r_corr"],
      "omega_corr": result["omega_corr"],
      "abs_lambda": result["abs_lambda"],
      "arg_lambda": result["arg_lambda"],
      "v0_tilde": result["v0_tilde"],
      "D_tilde": result["D_tilde"],
      "D0_lattice": result["D0_lattice"],
      "A": result["A"],
      "D_eff_cabp": result["D_eff_cabp"],
      "four_D_eff_cabp": result["four_D_eff_cabp"],
      "D_eff_cabp_corr": result["D_eff_cabp_corr"],
      "four_D_eff_cabp_corr": result["four_D_eff_cabp_corr"],
    })

  ax_msd.set_xlabel(r"$t$", fontsize=30)
  ax_msd.set_ylabel(r"$\mathrm{MSD}$", fontsize=32)
  ax_msd.set_xticks([1, 10, 100, 1000], labels=["1", "10", "100", "1000"], fontsize=28)
  ax_msd.set_yticks(
    [1, 10, 100, 1000, 10000],
    labels=["1", "10", "100", "1000", "10000"],
    fontsize=28,
  )
  y_min, y_max = ax_msd.get_ylim()
  ax_msd.set_ylim(y_min, y_max * 2.5)
  ax_msd.text(-0.12, 1.04, "(a)", transform=ax_msd.transAxes, fontsize=32)

  ax_vacf.set_xlabel(r"$t$", fontsize=30)
  ax_vacf.set_ylabel(r"$C_V^{\mathrm{diag}}$", fontsize=32)
  ax_vacf.set_xlim(1, T_MAX_PLOT)
  ax_vacf.set_xticks([20, 40, 60], labels=["20", "40", "60"], fontsize=28)
  y_ticks_vacf = ax_vacf.get_yticks()
  ax_vacf.set_yticks(y_ticks_vacf[::2], labels=[f"{y:.1f}" for y in y_ticks_vacf[::2]], fontsize=28)
  ax_vacf.text(-0.15, 1.04, "(b)", transform=ax_vacf.transAxes, fontsize=32)

  regime_handles = [
    Line2D([0], [0], marker="o", linestyle="none", color=color,
           markersize=10, label=label)
    for (label, _, _), color in zip(WALKERS, COLORS)
  ]
  mapping_handles = [
    Line2D([0], [0], marker="o", linestyle="none", color="black",
           markersize=10, label="aCRW"),
    Line2D([0], [0], linestyle="--", color="black", lw=lw_theory,
           label="cABP: weak-turning map"),
    Line2D([0], [0], linestyle=":", color="black", lw=lw_theory_corr,
           label="cABP: exact-correlation map"),
  ]
  ax_msd.legend(
    handles=regime_handles + mapping_handles,
    loc="lower right",
    frameon=False,
    fontsize=14,
    handlelength=2.3,
    labelspacing=0.3,
  )
  for ax in (ax_msd, ax_vacf):
    ax.tick_params(which="both", direction="in", top=True, right=True)
  fig_combined.tight_layout()

  SAVE_DIR.mkdir(parents=True, exist_ok=True)
  fig_combined.savefig(OUTPUT_COMBINED_PDF, format="pdf", dpi=300, bbox_inches="tight")
  fig_combined.savefig(OUTPUT_COMBINED_SVG, format="svg", dpi=300, bbox_inches="tight")
  plt.close(fig_combined)

  meta = {
    "figure": "figure5_cabp_vacf_msd",
    "N": N,
    "T": T,
    "DT_save": DT_save,
    "base_seed": SEED,
    "nu": NU,
    "ell": ELL,
    "n_per_decade": N_PER_DECADE,
    "linear_short_max": LINEAR_SHORT_MAX,
    "theory": "aCRW_markers_cABP_weak_turning_dashed_exact_correlation_dotted",
    "D_tilde": 0.0,
    "delta_t": DELTA_T,
    "walkers": meta_walkers,
    "output_combined_pdf": str(OUTPUT_COMBINED_PDF),
    "output_combined_svg": str(OUTPUT_COMBINED_SVG),
  }
  with open(OUTPUT_META, "w") as f:
    json.dump(meta, f, indent=2)

  print(f"Saved figure to {OUTPUT_COMBINED_PDF}")
  print(f"Saved figure to {OUTPUT_COMBINED_SVG}")
  print(f"Saved metadata to {OUTPUT_META}")


if __name__ == "__main__":
  main()
