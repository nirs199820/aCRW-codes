"""Analysis tools for random walk simulations."""

from __future__ import annotations

import numpy as np
from numba import njit, prange
import multiprocessing

from .evolution import Evolution, evolve
from .states import WalkState, get_probability_distribution


def msd_chunk(
  traj_x: list[int],
  traj_y: list[int],
  start_lag: int,
  end_lag: int,
  max_lag: int,
) -> np.ndarray:
  x = np.asarray(traj_x)
  y = np.asarray(traj_y)
  return np.array([
      (x[lag] - x[0]) ** 2 + (y[lag] - y[0]) ** 2
      for lag in range(start_lag, end_lag + 1)
  ])


def _msd_worker(args: tuple) -> np.ndarray:
  return msd_chunk(*args)


def msd_parallelize(
  traj_x: list[int],
  traj_y: list[int],
  max_lag: int,
  DT_save: int,
  n_workers: int = 4,
) -> np.ndarray:
  n_workers = min(n_workers, max_lag)
  max_lag_indx = max_lag // DT_save
  chunk_size = (max_lag_indx + n_workers - 1) // n_workers
  worker_args = [
    (traj_x, traj_y, start, min(start + chunk_size - 1, max_lag_indx), max_lag)
    for start in range(1, max_lag_indx + 1, chunk_size)
    ]
  with multiprocessing.Pool(n_workers) as pool:
    results = pool.map(_msd_worker, worker_args)
  return np.concatenate(results)

def calculate_msd(traj_x: list[int], traj_y: list[int], max_lag: int, DT_save: int) -> np.ndarray:
  return np.array([
      np.mean(
          (np.asarray(traj_x[:-lag]) - np.asarray(traj_x[lag:])) ** 2 +
          (np.asarray(traj_y[:-lag]) - np.asarray(traj_y[lag:])) ** 2
      )
      for lag in range(1, max_lag + 1, DT_save)
  ])

@njit(parallel=True)
def _calculate_msd_batched_numba(traj_x, traj_y, max_lag, dt_save, start_lag=1, jump_lag=1):
    n_particles, n_times = traj_x.shape
    lags = np.arange(start_lag, max_lag, jump_lag)
    msd = np.empty(lags.shape[0], dtype=np.float64)
    for i in prange(lags.shape[0]):
        lag = lags[i]
        total = 0.0
        count = 0
        for p in range(n_particles):
            for t in range(n_times - lag):
                dx = traj_x[p, t] - traj_x[p, t + lag]
                dy = traj_y[p, t] - traj_y[p, t + lag]
                total += dx * dx + dy * dy
                count += 1
        msd[i] = total / count
    return msd


def calculate_msd_batched_numba(traj_x, traj_y, max_lag, DT_save, start_lag=1, jump_lag=1):
    traj_x = np.ascontiguousarray(traj_x)
    traj_y = np.ascontiguousarray(traj_y)
    return _calculate_msd_batched_numba(traj_x, traj_y, max_lag, DT_save, start_lag, jump_lag)


@njit(parallel=True)
def _calculate_ensemble_msd_from_origin_batched_numba(traj_x, traj_y, max_lag, dt_save, start_lag=1, jump_lag=1):
    n_particles, n_times = traj_x.shape
    lags = np.arange(start_lag, max_lag, jump_lag)
    msd = np.empty(lags.shape[0], dtype=np.float64)
    for i in prange(lags.shape[0]):
        lag = lags[i]
        total = 0.0
        for p in range(n_particles):
            dx = traj_x[p, lag] - traj_x[p, 0]
            dy = traj_y[p, lag] - traj_y[p, 0]
            total += dx * dx + dy * dy
        msd[i] = total / n_particles
    return msd


def calculate_ensemble_msd_from_origin_batched_numba(traj_x, traj_y, max_lag, DT_save, start_lag=1, jump_lag=1):
    traj_x = np.ascontiguousarray(traj_x)
    traj_y = np.ascontiguousarray(traj_y)
    return _calculate_ensemble_msd_from_origin_batched_numba(traj_x, traj_y, max_lag, DT_save, start_lag, jump_lag)


@njit(parallel=True)
def _calculate_ensemble_msd_from_origin_at_lags_numba(traj_x, traj_y, lags):
    n_lags = lags.shape[0]
    n_particles = traj_x.shape[0]
    msd = np.empty(n_lags, dtype=np.float64)
    for i in prange(n_lags):
        lag = lags[i]
        total = 0.0
        for p in range(n_particles):
            dx = traj_x[p, lag] - traj_x[p, 0]
            dy = traj_y[p, lag] - traj_y[p, 0]
            total += dx * dx + dy * dy
        msd[i] = total / n_particles
    return msd


def calculate_ensemble_msd_from_origin_at_lags_numba(traj_x, traj_y, lags):
    """Ensemble MSD from origin evaluated at explicit lag indices."""
    traj_x = np.ascontiguousarray(traj_x)
    traj_y = np.ascontiguousarray(traj_y)
    lags = np.ascontiguousarray(np.asarray(lags, dtype=np.int64))
    return _calculate_ensemble_msd_from_origin_at_lags_numba(traj_x, traj_y, lags)


def msd_lag_schedule(
  T: int,
  *,
  lag_min: int = 1,
  lag_max: int | None = None,
  n_per_decade: int = 15,
  linear_short_max: int = 50,
) -> np.ndarray:
  """Build hybrid MSD lag schedule: dense at short times, log-spaced at long times."""
  if lag_max is None:
    lag_max = T - 1
  lag_max = min(lag_max, T - 1)
  if lag_max < lag_min:
    return np.array([], dtype=np.int64)

  lags_lin = np.arange(lag_min, min(linear_short_max, lag_max) + 1, dtype=np.int64)
  log_start = max(lag_min, min(linear_short_max, lag_max))
  if log_start < lag_max:
    n_log = max(2, int(np.ceil(np.log10(lag_max / log_start) * n_per_decade)))
    lags_log = np.round(
      np.logspace(np.log10(log_start), np.log10(lag_max), n_log)
    ).astype(np.int64)
    lags = np.unique(np.concatenate([lags_lin, lags_log]))
  else:
    lags = lags_lin

  return lags[(lags >= lag_min) & (lags <= lag_max)]


def fit_msd_power_law(
  times: np.ndarray,
  msd: np.ndarray,
  t_fit_min: float,
  t_fit_max: float,
) -> tuple[np.ndarray, np.ndarray]:
  """Fit log MSD ~ m log t + b over a physical-time window."""
  times = np.asarray(times, dtype=np.float64)
  msd = np.asarray(msd, dtype=np.float64)
  fit_mask = (times >= t_fit_min) & (times <= t_fit_max) & (msd > 0) & (times > 0)
  if np.count_nonzero(fit_mask) < 2:
    return np.array([np.nan, np.nan], dtype=np.float64), fit_mask
  coeffs = np.polyfit(np.log(times[fit_mask]), np.log(msd[fit_mask]), 1)
  return coeffs, fit_mask


def plot_msd_loglog(
  times: np.ndarray,
  msd: np.ndarray,
  coeffs: np.ndarray,
  fit_mask: np.ndarray,
  *,
  title: str = "MSD",
  expected_D: float | None = None,
  show_diffusion_reference: bool = True,
):
  """Plot MSD on log-log axes with power-law fit over the selected window."""
  import matplotlib.pyplot as plt

  times = np.asarray(times, dtype=np.float64)
  msd = np.asarray(msd, dtype=np.float64)
  fig, ax = plt.subplots()
  ax.loglog(times, msd, "o-", ms=3, label="MSD")
  if np.any(fit_mask):
    ax.loglog(times[fit_mask], msd[fit_mask], "o", ms=5, label="fit window", alpha=0.8)
    ax.loglog(
      times,
      np.exp(coeffs[1]) * times ** coeffs[0],
      "--",
      label=f"fit: m={coeffs[0]:.2f}, D_active={np.exp(coeffs[1]) / 4:.2f}",
    )
  if expected_D is not None:
    ax.loglog(
      times,
      4 * expected_D * times,
      "--",
      label=f"theory: MSD=4Dt, D={expected_D:.2f}",
    )
  elif show_diffusion_reference:
    ax.loglog(times, times, "--", label="diffusion (m=1)")
  ax.set_xlabel("time")
  ax.set_ylabel("MSD")
  ax.legend()
  fig.canvas.manager.set_window_title(title)
  return fig


@njit(parallel=True)
def _calculate_velocity_autocorr_batched_numba(dir_x, dir_y, n_lags):
    n_particles, n_times = dir_x.shape
    n_corr = n_times - (n_lags + 1)
    dx_corr = np.empty(n_lags, dtype=np.float64)
    dy_corr = np.empty(n_lags, dtype=np.float64)
    for lag in prange(n_lags):
        total_x = 0.0
        total_y = 0.0
        for p in range(n_particles):
            for t in range(n_corr):
                total_x += dir_x[p, t] * dir_x[p, t + lag]
                total_y += dir_y[p, t] * dir_y[p, t + lag]
        count = n_particles * n_corr
        dx_corr[lag] = total_x / count
        dy_corr[lag] = total_y / count
    return dx_corr, dy_corr


def calculate_velocity_autocorr_batched_numba(dir_x, dir_y, n_lags):
    dir_x = np.ascontiguousarray(dir_x)
    dir_y = np.ascontiguousarray(dir_y)
    return _calculate_velocity_autocorr_batched_numba(dir_x, dir_y, n_lags)


@njit(parallel=True)
def _calculate_direction_crosscorr_batched_numba(dir_x, dir_y, n_lags):
    n_particles, n_times = dir_x.shape
    n_corr = n_times - (n_lags + 1)
    c_parallel = np.empty(n_lags, dtype=np.float64)
    c_perp = np.empty(n_lags, dtype=np.float64)
    for lag in prange(n_lags):
        total_parallel = 0.0
        total_perp = 0.0
        for p in range(n_particles):
            for t in range(n_corr):
                dx0 = dir_x[p, t]
                dy0 = dir_y[p, t]
                dx1 = dir_x[p, t + lag]
                dy1 = dir_y[p, t + lag]
                total_parallel += dx0 * dx1 + dy0 * dy1
                total_perp += dx0 * dy1 - dy0 * dx1
        count = n_particles * n_corr
        c_parallel[lag] = total_parallel / count
        c_perp[lag] = total_perp / count
    return c_parallel, c_perp


def calculate_direction_crosscorr_batched_numba(dir_x, dir_y, n_lags):
    """Ensemble-averaged direction cross-correlations from saved direction components."""
    dir_x = np.ascontiguousarray(dir_x)
    dir_y = np.ascontiguousarray(dir_y)
    return _calculate_direction_crosscorr_batched_numba(dir_x, dir_y, n_lags)


@njit(parallel=True)
def _calculate_velocity_autocorr_batched_at_lags_numba(dir_x, dir_y, lags):
    n_particles, n_times = dir_x.shape
    n_out = lags.shape[0]
    dx_corr = np.empty(n_out, dtype=np.float64)
    dy_corr = np.empty(n_out, dtype=np.float64)
    for i in prange(n_out):
        lag = lags[i]
        n_corr = n_times - lag - 1
        if n_corr <= 0:
            dx_corr[i] = np.nan
            dy_corr[i] = np.nan
            continue
        total_x = 0.0
        total_y = 0.0
        for p in range(n_particles):
            for t in range(n_corr):
                total_x += dir_x[p, t] * dir_x[p, t + lag]
                total_y += dir_y[p, t] * dir_y[p, t + lag]
        count = n_particles * n_corr
        dx_corr[i] = total_x / count
        dy_corr[i] = total_y / count
    return dx_corr, dy_corr


def calculate_velocity_autocorr_batched_at_lags_numba(dir_x, dir_y, lags):
    """Ensemble-averaged velocity autocorrelations at selected lags."""
    dir_x = np.ascontiguousarray(dir_x)
    dir_y = np.ascontiguousarray(dir_y)
    lags = np.ascontiguousarray(np.asarray(lags, dtype=np.int64))
    return _calculate_velocity_autocorr_batched_at_lags_numba(dir_x, dir_y, lags)


@njit(parallel=True)
def _calculate_direction_crosscorr_batched_at_lags_numba(dir_x, dir_y, lags):
    n_particles, n_times = dir_x.shape
    n_out = lags.shape[0]
    c_parallel = np.empty(n_out, dtype=np.float64)
    c_perp = np.empty(n_out, dtype=np.float64)
    for i in prange(n_out):
        lag = lags[i]
        n_corr = n_times - lag - 1
        if n_corr <= 0:
            c_parallel[i] = np.nan
            c_perp[i] = np.nan
            continue
        total_parallel = 0.0
        total_perp = 0.0
        for p in range(n_particles):
            for t in range(n_corr):
                dx0 = dir_x[p, t]
                dy0 = dir_y[p, t]
                dx1 = dir_x[p, t + lag]
                dy1 = dir_y[p, t + lag]
                total_parallel += dx0 * dx1 + dy0 * dy1
                total_perp += dx0 * dy1 - dy0 * dx1
        count = n_particles * n_corr
        c_parallel[i] = total_parallel / count
        c_perp[i] = total_perp / count
    return c_parallel, c_perp


def calculate_direction_crosscorr_batched_at_lags_numba(dir_x, dir_y, lags):
    """Ensemble-averaged direction cross-correlations at selected lags."""
    dir_x = np.ascontiguousarray(dir_x)
    dir_y = np.ascontiguousarray(dir_y)
    lags = np.ascontiguousarray(np.asarray(lags, dtype=np.int64))
    return _calculate_direction_crosscorr_batched_at_lags_numba(dir_x, dir_y, lags)


def calculate_msd_batched(
  traj_x: np.ndarray,
  traj_y: np.ndarray,
  max_lag: int,
  DT_save: int,
) -> np.ndarray:
  traj_x = np.asarray(traj_x)
  traj_y = np.asarray(traj_y)
  return np.array([
      np.mean(
          (traj_x[:, :-lag] - traj_x[:, lag:]) ** 2 +
          (traj_y[:, :-lag] - traj_y[:, lag:]) ** 2
      )
      for lag in range(1, max_lag + 1, DT_save)
  ])


def calculate_ensemble_msd_from_origin_batched(
  traj_x: np.ndarray,
  traj_y: np.ndarray,
  max_lag: int,
  DT_save: int,
  start_lag: int = 1,
  jump_lag: int = 1,
) -> np.ndarray:
  traj_x = np.asarray(traj_x)
  traj_y = np.asarray(traj_y)
  return np.array([
      np.mean(
          (traj_x[:, lag] - traj_x[:, 0]) ** 2 +
          (traj_y[:, lag] - traj_y[:, 0]) ** 2
      )
      for lag in range(start_lag, max_lag + 1, jump_lag)
  ])


def tamsd_chunk(
  traj_x: list[int],
  traj_y: list[int],
  start_lag: int,
  end_lag: int,
  max_lag: int,
) -> np.ndarray:
  return np.array([
      np.mean(
          (np.asarray(traj_x[:-lag]) - np.asarray(traj_x[lag:])) ** 2 +
          (np.asarray(traj_y[:-lag]) - np.asarray(traj_y[lag:])) ** 2
      )
      for lag in range(start_lag, end_lag + 1)
  ])


def _tamsd_worker(args: tuple) -> np.ndarray:
  return tamsd_chunk(*args)


def tamsd_parallelize(
  traj_x: list[int],
  traj_y: list[int],
  max_lag: int,
  n_workers: int = 4,
) -> np.ndarray:
  n_workers = min(n_workers, max_lag)
  chunk_size = (max_lag + n_workers - 1) // n_workers
  worker_args = [
    (traj_x, traj_y, start, min(start + chunk_size - 1, max_lag), max_lag)
    for start in range(1, max_lag + 1, chunk_size)
    ]
  with multiprocessing.Pool(n_workers) as pool:
    results = pool.map(_tamsd_worker, worker_args)
  return np.concatenate(results)


def compute_msd(
  initial_state: WalkState,
  evolution: Evolution,
  max_time: int,
) -> np.ndarray:
  state = WalkState(initial_state.probabilities.copy(), initial_state.grid_size)
  msd_values = np.zeros(max_time)
  d = state.grid_size
  center = d / 2

  for t in range(max_time):
    evolve(state, evolution)

    prob_dist = get_probability_distribution(state)
    msd = 0.0

    for x in range(d):
      for y in range(d):
        dx = x - center
        dy = y - center
        msd += prob_dist[x, y] * (dx ** 2 + dy ** 2)

    msd_values[t] = msd

  return msd_values


_EPS = 1e-15
_ANALYSIS_DTYPE = np.float32

_SHAPE_METRIC_KEYS = (
  "end_to_end_sq",
  "end_to_end",
  "rg_sq",
  "lambda1",
  "lambda2",
  "aspect_ratio",
  "asphericity",
  "d_eff",
  "path_length",
  "tortuosity",
  "n_unique_sites",
  "revisit_fraction",
  "compactness_rg",
  "signed_area",
  "signed_area_rg",
  "hull_area",
  "compactness_hull",
  "signed_area_hull",
)

_HULL_METRIC_KEYS = ("hull_area", "compactness_hull", "signed_area_hull")

SHAPE_METRIC_DISTRIBUTION_KEYS = (
  "rg_sq",
  "end_to_end_sq",
  "aspect_ratio",
  "asphericity",
  "d_eff",
  "tortuosity",
  "revisit_fraction",
  "compactness_rg",
  "D_occ",
)

_METRIC_PLOT_LABELS = {
  "rg_sq": "R_g",
  "end_to_end_sq": "end-to-end^2",
  "aspect_ratio": "aspect ratio",
  "asphericity": "asphericity",
  "d_eff": "d_eff",
  "tortuosity": "tortuosity",
  "revisit_fraction": "revisit fraction",
  "compactness_rg": "compactness (R_g)",
  "D_occ": "D_occ",
}


def _nan_shape_metrics(*, include_hull: bool = False) -> dict:
  metrics = {key: np.nan for key in _SHAPE_METRIC_KEYS if not key.startswith("hull")}
  if include_hull:
    metrics["hull_area"] = np.nan
    metrics["compactness_hull"] = np.nan
    metrics["signed_area_hull"] = np.nan
  return metrics


def _summarize_metric(values: np.ndarray, *, include_per_trajectory: bool) -> dict:
  values = np.asarray(values, dtype=_ANALYSIS_DTYPE)
  if values.size == 0 or np.all(np.isnan(values)):
    summary = {
      "mean": None,
      "median": None,
      "std": None,
      "p25": None,
      "p75": None,
    }
  else:
    summary = {
      "mean": float(np.nanmean(values)),
      "median": float(np.nanmedian(values)),
      "std": float(np.nanstd(values)),
      "p25": float(np.nanpercentile(values, 25)),
      "p75": float(np.nanpercentile(values, 75)),
    }
  if include_per_trajectory:
    summary["per_trajectory"] = values
  return summary


def _unpack_trajectory_inputs(traj_x, traj_y=None):
  if traj_y is None:
    traj_x = np.asarray(traj_x)
    if traj_x.ndim == 3 and traj_x.shape[2] == 2:
      return traj_x[:, :, 0], traj_x[:, :, 1]
    raise ValueError("Provide traj_x and traj_y with shape (N, T), or stacked traj_x with shape (N, T, 2)")
  return np.asarray(traj_x), np.asarray(traj_y)


def trajectory_shape_metrics_xy(
  traj_x: np.ndarray,
  traj_y: np.ndarray,
  *,
  lattice: bool = True,
  compute_hull: bool = False,
) -> dict:
  """Compute geometric shape observables from separate x/y trajectory arrays.

  Integer lattice coordinates are promoted to float32 only for metric math.
  """
  traj_x = np.asarray(traj_x)
  traj_y = np.asarray(traj_y)
  if traj_x.shape != traj_y.shape or traj_x.ndim != 1:
    raise ValueError("traj_x and traj_y must be 1D arrays of equal length")

  n_points = traj_x.shape[0]
  if n_points < 1:
    return _nan_shape_metrics(include_hull=compute_hull)

  x = traj_x.astype(_ANALYSIS_DTYPE, copy=False)
  y = traj_y.astype(_ANALYSIS_DTYPE, copy=False)

  disp_x = x[-1] - x[0]
  disp_y = y[-1] - y[0]
  end_to_end_sq = float(disp_x * disp_x + disp_y * disp_y)
  end_to_end = float(np.sqrt(end_to_end_sq))

  if n_points < 2:
    metrics = _nan_shape_metrics(include_hull=compute_hull)
    metrics["end_to_end_sq"] = end_to_end_sq
    metrics["end_to_end"] = end_to_end
    return metrics

  r_cm_x = float(x.mean())
  r_cm_y = float(y.mean())
  centered_x = x - r_cm_x
  centered_y = y - r_cm_y
  rg_sq = float(np.mean(centered_x ** 2 + centered_y ** 2))

  g_xx = float(np.mean(centered_x * centered_x))
  g_xy = float(np.mean(centered_x * centered_y))
  g_yy = float(np.mean(centered_y * centered_y))
  trace = g_xx + g_yy
  det = g_xx * g_yy - g_xy * g_xy
  disc = max(trace * trace - 4.0 * det, 0.0)
  lambda1 = float(0.5 * (trace + np.sqrt(disc)))
  lambda2 = float(0.5 * (trace - np.sqrt(disc)))

  aspect_ratio = float(np.sqrt(lambda1 / max(lambda2, _EPS)))
  asphericity = float(((lambda1 - lambda2) ** 2) / max((lambda1 + lambda2) ** 2, _EPS))
  d_eff = float((lambda1 + lambda2) ** 2 / max(lambda1 ** 2 + lambda2 ** 2, _EPS))

  step_x = np.diff(x)
  step_y = np.diff(y)
  step_lengths = np.sqrt(step_x ** 2 + step_y ** 2)
  path_length = float(np.sum(step_lengths))
  tortuosity = float(path_length / max(end_to_end, _EPS))

  if lattice:
    sites = np.unique(np.column_stack([traj_x, traj_y]), axis=0)
    n_unique_sites = int(sites.shape[0])
    revisit_fraction = float(1.0 - n_unique_sites / n_points)
    compactness_rg = float(n_unique_sites / max(np.pi * rg_sq, _EPS))
    unique_sites = sites
  else:
    n_unique_sites = np.nan
    revisit_fraction = np.nan
    compactness_rg = np.nan
    unique_sites = None

  signed_area = float(
    0.5 * np.sum(x[:-1] * y[1:] - y[:-1] * x[1:])
  )
  signed_area_rg = float(signed_area / max(np.pi * rg_sq, _EPS))

  hull_area = np.nan
  compactness_hull = np.nan
  signed_area_hull = np.nan
  if compute_hull and lattice and unique_sites is not None and unique_sites.shape[0] >= 3:
    try:
      from scipy.spatial import ConvexHull

      hull = ConvexHull(unique_sites)
      hull_area = float(hull.volume)
      compactness_hull = float(n_unique_sites / max(hull_area, _EPS))
      signed_area_hull = float(signed_area / max(hull_area, _EPS))
    except Exception:
      pass

  return {
    "end_to_end_sq": end_to_end_sq,
    "end_to_end": end_to_end,
    "rg_sq": rg_sq,
    "lambda1": lambda1,
    "lambda2": lambda2,
    "aspect_ratio": aspect_ratio,
    "asphericity": asphericity,
    "d_eff": d_eff,
    "path_length": path_length,
    "tortuosity": tortuosity,
    "n_unique_sites": n_unique_sites,
    "revisit_fraction": revisit_fraction,
    "compactness_rg": compactness_rg,
    "signed_area": signed_area,
    "signed_area_rg": signed_area_rg,
    "hull_area": hull_area,
    "compactness_hull": compactness_hull,
    "signed_area_hull": signed_area_hull,
  }


def trajectory_shape_metrics(
  traj: np.ndarray,
  *,
  lattice: bool = True,
  compute_hull: bool = False,
) -> dict:
  """Compute geometric shape observables for a single 2D trajectory (T, 2)."""
  traj = np.asarray(traj)
  if traj.ndim != 2 or traj.shape[1] != 2:
    raise ValueError("traj must have shape (T, 2)")
  return trajectory_shape_metrics_xy(
    traj[:, 0], traj[:, 1], lattice=lattice, compute_hull=compute_hull
  )


def batch_trajectory_shape_metrics(
  traj_x,
  traj_y=None,
  *,
  lattice: bool = True,
  compute_hull: bool = False,
  include_per_trajectory: bool = True,
) -> dict:
  """Compute shape observables for an ensemble of 2D trajectories.

  Primary usage: traj_x, traj_y with shape (N, T). Also accepts stacked (N, T, 2).
  """
  traj_x, traj_y = _unpack_trajectory_inputs(traj_x, traj_y)
  if traj_x.ndim != 2 or traj_y.ndim != 2 or traj_x.shape != traj_y.shape:
    raise ValueError("traj_x and traj_y must have shape (N, T)")

  n_particles = traj_x.shape[0]
  metric_keys = [key for key in _SHAPE_METRIC_KEYS if compute_hull or key not in _HULL_METRIC_KEYS]
  values_by_key = {key: np.empty(n_particles, dtype=_ANALYSIS_DTYPE) for key in metric_keys}

  for i in range(n_particles):
    metrics = trajectory_shape_metrics_xy(
      traj_x[i], traj_y[i], lattice=lattice, compute_hull=compute_hull
    )
    for key in metric_keys:
      values_by_key[key][i] = metrics[key]

  return {
    key: _summarize_metric(values, include_per_trajectory=include_per_trajectory)
    for key, values in values_by_key.items()
  }


def per_trajectory_rg_sq(traj_x, traj_y=None) -> np.ndarray:
  """Per-trajectory radius of gyration squared over the full saved trajectory."""
  return per_trajectory_shape_values(traj_x, traj_y, ("rg_sq",))["rg_sq"]


def per_trajectory_shape_values(
  traj_x,
  traj_y=None,
  metric_keys: tuple[str, ...] | list[str] = SHAPE_METRIC_DISTRIBUTION_KEYS,
  *,
  lattice: bool = True,
  compute_hull: bool = False,
) -> dict[str, np.ndarray]:
  """Extract selected per-trajectory shape metric arrays for an ensemble."""
  traj_x, traj_y = _unpack_trajectory_inputs(traj_x, traj_y)
  if traj_x.ndim != 2 or traj_y.ndim != 2 or traj_x.shape != traj_y.shape:
    raise ValueError("traj_x and traj_y must have shape (N, T)")

  n_particles = traj_x.shape[0]
  values_by_key = {
    key: np.empty(n_particles, dtype=_ANALYSIS_DTYPE) for key in metric_keys
  }
  for i in range(n_particles):
    metrics = trajectory_shape_metrics_xy(
      traj_x[i], traj_y[i], lattice=lattice, compute_hull=compute_hull
    )
    for key in metric_keys:
      values_by_key[key][i] = metrics[key]
  return values_by_key


def occupation_prefix_schedule(
  T: int,
  *,
  lag_min: int = 1,
  n_per_decade: int = 15,
  linear_short_max: int = 50,
) -> np.ndarray:
  """Prefix lengths for growing-occupation analysis, including the full trajectory."""
  prefixes = msd_lag_schedule(
    T,
    lag_min=lag_min,
    lag_max=T,
    n_per_decade=n_per_decade,
    linear_short_max=linear_short_max,
  )
  if prefixes.size == 0 or prefixes[-1] != T:
    prefixes = np.unique(np.append(prefixes, T))
  return prefixes.astype(np.int64)


def occupation_rg_and_unique_sites_at_prefixes(
  traj_x: np.ndarray,
  traj_y: np.ndarray,
  prefix_lengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Compute R_g(T) and S(T) for growing trajectory prefixes.

  S(T) is the number of unique visited lattice sites up to time T.
  R_g(T) is the radius of gyration of all positions up to time T.
  """
  traj_x = np.asarray(traj_x)
  traj_y = np.asarray(traj_y)
  if traj_x.shape != traj_y.shape or traj_x.ndim != 1:
    raise ValueError("traj_x and traj_y must be 1D arrays of equal length")

  prefix_lengths = np.asarray(prefix_lengths, dtype=np.int64)
  if prefix_lengths.size == 0:
    return np.array([], dtype=_ANALYSIS_DTYPE), np.array([], dtype=_ANALYSIS_DTYPE)

  n_points = traj_x.shape[0]
  prefix_lengths = prefix_lengths[(prefix_lengths >= 1) & (prefix_lengths <= n_points)]
  if prefix_lengths.size == 0:
    return np.array([], dtype=_ANALYSIS_DTYPE), np.array([], dtype=_ANALYSIS_DTYPE)

  r_g = np.empty(prefix_lengths.size, dtype=_ANALYSIS_DTYPE)
  unique_sites = np.empty(prefix_lengths.size, dtype=_ANALYSIS_DTYPE)

  visited: set[tuple[int, int]] = set()
  schedule_idx = 0
  sum_x = 0.0
  sum_y = 0.0
  sum_x2 = 0.0
  sum_y2 = 0.0

  for t in range(n_points):
    x = float(traj_x[t])
    y = float(traj_y[t])
    visited.add((int(traj_x[t]), int(traj_y[t])))
    n = t + 1
    sum_x += x
    sum_y += y
    sum_x2 += x * x
    sum_y2 += y * y

    while schedule_idx < prefix_lengths.size and n == prefix_lengths[schedule_idx]:
      cm_x = sum_x / n
      cm_y = sum_y / n
      rg_sq = (sum_x2 + sum_y2) / n - cm_x * cm_x - cm_y * cm_y
      r_g[schedule_idx] = float(np.sqrt(max(rg_sq, 0.0)))
      unique_sites[schedule_idx] = float(len(visited))
      schedule_idx += 1

  return r_g, unique_sites


def fit_occupation_dimension(
  prefix_times: np.ndarray,
  r_g: np.ndarray,
  unique_sites: np.ndarray,
  t_fit_min: float,
  t_fit_max: float,
) -> tuple[float, np.ndarray]:
  """Fit D_occ = d log R_g / d log S over a physical-time window.

  The scaling ansatz S ~ R_g^(1 / D_occ) is equivalent to this slope.
  """
  prefix_times = np.asarray(prefix_times, dtype=np.float64)
  r_g = np.asarray(r_g, dtype=np.float64)
  unique_sites = np.asarray(unique_sites, dtype=np.float64)
  fit_mask = (
    (prefix_times >= t_fit_min)
    & (prefix_times <= t_fit_max)
    & (r_g > 0)
    & (unique_sites > 1)
  )
  if np.count_nonzero(fit_mask) < 2:
    return np.nan, fit_mask
  coeffs = np.polyfit(
    np.log(unique_sites[fit_mask]),
    np.log(r_g[fit_mask]),
    1,
  )
  return float(coeffs[0]), fit_mask


def occupation_dimension_xy(
  traj_x: np.ndarray,
  traj_y: np.ndarray,
  prefix_lengths: np.ndarray,
  t_fit_min: float,
  t_fit_max: float,
  *,
  dt_save: float = 1.0,
) -> dict:
  """Compute occupation dimension and prefix curves for one trajectory."""
  r_g, unique_sites = occupation_rg_and_unique_sites_at_prefixes(
    traj_x, traj_y, prefix_lengths
  )
  prefix_times = np.asarray(prefix_lengths, dtype=np.float64) * dt_save
  d_occ, fit_mask = fit_occupation_dimension(
    prefix_times, r_g, unique_sites, t_fit_min, t_fit_max
  )
  return {
    "D_occ": d_occ,
    "r_g": r_g,
    "unique_sites": unique_sites,
    "prefix_times": prefix_times,
    "fit_mask": fit_mask,
  }


def batch_occupation_dimension(
  traj_x,
  traj_y=None,
  *,
  prefix_lengths: np.ndarray,
  t_fit_min: float,
  t_fit_max: float,
  dt_save: float = 1.0,
  include_per_trajectory: bool = True,
) -> dict:
  """Compute D_occ for an ensemble of trajectories."""
  traj_x, traj_y = _unpack_trajectory_inputs(traj_x, traj_y)
  if traj_x.ndim != 2 or traj_y.ndim != 2 or traj_x.shape != traj_y.shape:
    raise ValueError("traj_x and traj_y must have shape (N, T)")

  n_particles = traj_x.shape[0]
  d_occ_values = np.empty(n_particles, dtype=_ANALYSIS_DTYPE)
  prefix_times = np.asarray(prefix_lengths, dtype=np.float64) * dt_save

  for i in range(n_particles):
    r_g, unique_sites = occupation_rg_and_unique_sites_at_prefixes(
      traj_x[i], traj_y[i], prefix_lengths
    )
    d_occ_values[i], _ = fit_occupation_dimension(
      prefix_times, r_g, unique_sites, t_fit_min, t_fit_max
    )

  return {
    "D_occ": _summarize_metric(d_occ_values, include_per_trajectory=include_per_trajectory),
  }


def plot_occupation_scaling(
  unique_sites: np.ndarray,
  r_g: np.ndarray,
  d_occ: float,
  fit_mask: np.ndarray,
  *,
  title: str = "Occupation scaling",
):
  """Plot log R_g vs log S with the fitted occupation dimension."""
  import matplotlib.pyplot as plt

  unique_sites = np.asarray(unique_sites, dtype=np.float64)
  r_g = np.asarray(r_g, dtype=np.float64)
  fit_mask = np.asarray(fit_mask, dtype=bool)

  fig, ax = plt.subplots()
  valid = (unique_sites > 0) & (r_g > 0)
  ax.loglog(unique_sites[valid], r_g[valid], "o-", ms=3, label=r"$R_g(S)$")
  if np.any(fit_mask) and np.isfinite(d_occ):
    s_fit = unique_sites[fit_mask]
    r_fit = r_g[fit_mask]
    ax.loglog(s_fit, r_fit, "o", ms=5, label="fit window", alpha=0.8)
    log_prefactor = np.polyfit(np.log(s_fit), np.log(r_fit), 1)[1]
    ax.loglog(
      s_fit,
      np.exp(log_prefactor) * s_fit ** d_occ,
      "--",
      label=rf"$D_{{occ}}={d_occ:.2f}$",
    )
  ax.set_xlabel("S (unique sites)")
  ax.set_ylabel(r"$R_g$")
  ax.legend()
  fig.canvas.manager.set_window_title(title)
  return fig


def plot_occupation_dimension_distribution(
  d_occ_values: np.ndarray,
  *,
  bins: int = 100,
):
  """Plot the ensemble distribution of occupation dimension D_occ."""
  import matplotlib.pyplot as plt

  values = np.asarray(d_occ_values, dtype=_ANALYSIS_DTYPE)
  finite = values[np.isfinite(values)]

  fig, ax = plt.subplots()
  if finite.size > 0:
    ax.hist(finite, bins=bins, density=True)
  ax.set_xlabel(r"$D_{occ}$")
  ax.set_ylabel("density")
  fig.canvas.manager.set_window_title("Occupation dimension distribution")
  return fig


def _metric_values_for_plot(metric_key: str, values: np.ndarray) -> np.ndarray:
  values = np.asarray(values, dtype=_ANALYSIS_DTYPE)
  if metric_key == "rg_sq":
    return np.sqrt(values)
  return values


def velocity_cross_correlations_xy(
  traj_x: np.ndarray,
  traj_y: np.ndarray,
  n_lags: int,
) -> dict:
  """Compute parallel and perpendicular velocity cross-correlations from x/y arrays."""
  traj_x = np.asarray(traj_x)
  traj_y = np.asarray(traj_y)
  if traj_x.shape != traj_y.shape or traj_x.ndim != 1:
    raise ValueError("traj_x and traj_y must be 1D arrays of equal length")

  vx = np.diff(traj_x)
  vy = np.diff(traj_y)
  n_vel = vx.shape[0]
  n_lags = min(int(n_lags), max(n_vel, 0))
  lags = np.arange(n_lags, dtype=int)
  c_parallel = np.full(n_lags, np.nan, dtype=_ANALYSIS_DTYPE)
  c_perp = np.full(n_lags, np.nan, dtype=_ANALYSIS_DTYPE)

  if n_vel < 1 or n_lags < 1:
    return {"lags": lags, "C_parallel": c_parallel, "C_perp": c_perp}

  for lag in range(n_lags):
    if lag == 0:
      vx0, vy0 = vx, vy
      vx1, vy1 = vx, vy
    else:
      vx0, vy0 = vx[:-lag], vy[:-lag]
      vx1, vy1 = vx[lag:], vy[lag:]
    if vx0.shape[0] == 0:
      continue
    c_parallel[lag] = np.mean(vx0 * vx1 + vy0 * vy1, dtype=_ANALYSIS_DTYPE)
    c_perp[lag] = np.mean(vx0 * vy1 - vy0 * vx1, dtype=_ANALYSIS_DTYPE)

  return {"lags": lags, "C_parallel": c_parallel, "C_perp": c_perp}


def velocity_cross_correlations(traj: np.ndarray, n_lags: int) -> dict:
  """Compute velocity cross-correlations for a single trajectory with shape (T, 2)."""
  traj = np.asarray(traj)
  if traj.ndim != 2 or traj.shape[1] != 2:
    raise ValueError("traj must have shape (T, 2)")
  return velocity_cross_correlations_xy(traj[:, 0], traj[:, 1], n_lags)


def batch_velocity_cross_correlations(
  traj_x,
  traj_y=None,
  n_lags: int = 0,
  *,
  include_per_trajectory: bool = False,
) -> dict:
  """Compute ensemble-averaged velocity cross-correlations from (N, T) x/y arrays."""
  if isinstance(traj_y, (int, np.integer)):
    n_lags = int(traj_y)
    traj_y = None
  traj_x, traj_y = _unpack_trajectory_inputs(traj_x, traj_y)
  if traj_x.ndim != 2 or traj_y.ndim != 2 or traj_x.shape != traj_y.shape:
    raise ValueError("traj_x and traj_y must have shape (N, T)")

  vx = np.diff(traj_x, axis=1)
  vy = np.diff(traj_y, axis=1)
  n_vel = vx.shape[1]
  n_lags = min(int(n_lags), max(n_vel, 0))
  lags = np.arange(n_lags, dtype=int)
  c_parallel = np.full(n_lags, np.nan, dtype=_ANALYSIS_DTYPE)
  c_perp = np.full(n_lags, np.nan, dtype=_ANALYSIS_DTYPE)

  if n_vel < 1 or n_lags < 1:
    result = {"lags": lags, "C_parallel": c_parallel, "C_perp": c_perp}
    if include_per_trajectory:
      result["C_parallel_per_trajectory"] = np.empty((traj_x.shape[0], 0), dtype=_ANALYSIS_DTYPE)
      result["C_perp_per_trajectory"] = np.empty((traj_x.shape[0], 0), dtype=_ANALYSIS_DTYPE)
    return result

  for lag in range(n_lags):
    if lag == 0:
      vx0, vy0 = vx, vy
      vx1, vy1 = vx, vy
    else:
      vx0, vy0 = vx[:, :-lag], vy[:, :-lag]
      vx1, vy1 = vx[:, lag:], vy[:, lag:]
    dot = vx0 * vx1 + vy0 * vy1
    cross = vx0 * vy1 - vy0 * vx1
    c_parallel[lag] = np.mean(dot, dtype=_ANALYSIS_DTYPE)
    c_perp[lag] = np.mean(cross, dtype=_ANALYSIS_DTYPE)

  result = {"lags": lags, "C_parallel": c_parallel, "C_perp": c_perp}
  if include_per_trajectory:
    per_parallel = np.full((traj_x.shape[0], n_lags), np.nan, dtype=_ANALYSIS_DTYPE)
    per_perp = np.full((traj_x.shape[0], n_lags), np.nan, dtype=_ANALYSIS_DTYPE)
    for i in range(traj_x.shape[0]):
      single = velocity_cross_correlations_xy(traj_x[i], traj_y[i], n_lags)
      per_parallel[i] = single["C_parallel"]
      per_perp[i] = single["C_perp"]
    result["C_parallel_per_trajectory"] = per_parallel
    result["C_perp_per_trajectory"] = per_perp
  return result


def box_counting_dimension(
  traj_x,
  traj_y=None,
  eps_list: np.ndarray | None = None,
) -> dict:
  """Estimate the box-counting dimension of a 2D trajectory trace.

  Accepts either shape (T, 2) or separate 1D x/y arrays.
  """
  if traj_y is None:
    traj = np.asarray(traj_x, dtype=_ANALYSIS_DTYPE)
    if traj.ndim != 2 or traj.shape[1] != 2:
      raise ValueError("traj must have shape (T, 2)")
    x = traj[:, 0]
    y = traj[:, 1]
  else:
    x = np.asarray(traj_x, dtype=_ANALYSIS_DTYPE)
    y = np.asarray(traj_y, dtype=_ANALYSIS_DTYPE)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
      raise ValueError("traj_x and traj_y must be 1D arrays of equal length")

  if x.shape[0] < 2:
    return {"box_sizes": np.array([]), "box_counts": np.array([]), "D_box": np.nan}

  if eps_list is None:
    eps_list = np.logspace(0, 2, 15)
  eps_list = np.asarray(eps_list, dtype=_ANALYSIS_DTYPE)
  eps_list = eps_list[eps_list > 0]
  if eps_list.size == 0:
    return {"box_sizes": eps_list, "box_counts": np.array([]), "D_box": np.nan}

  shifted_x = x - x.min()
  shifted_y = y - y.min()
  box_counts = np.empty(eps_list.size, dtype=_ANALYSIS_DTYPE)
  for i, eps in enumerate(eps_list):
    box_indices = np.column_stack([
      np.floor(shifted_x / eps).astype(np.int32),
      np.floor(shifted_y / eps).astype(np.int32),
    ])
    box_counts[i] = len({tuple(idx) for idx in box_indices})

  valid = box_counts > 0
  if np.count_nonzero(valid) < 2:
    return {"box_sizes": eps_list, "box_counts": box_counts, "D_box": np.nan}

  slope, _ = np.polyfit(
    np.log(1.0 / eps_list[valid]),
    np.log(box_counts[valid]),
    1,
  )
  return {"box_sizes": eps_list, "box_counts": box_counts, "D_box": float(slope)}


def box_counting_dimension_at_lags(
  traj_x: np.ndarray,
  traj_y: np.ndarray,
  time_lags: np.ndarray,
  *,
  eps_list: np.ndarray | None = None,
) -> dict:
  """Estimate box-counting dimension from trajectory points at selected times."""
  traj_x = np.asarray(traj_x)
  traj_y = np.asarray(traj_y)
  if traj_x.shape != traj_y.shape or traj_x.ndim != 1:
    raise ValueError("traj_x and traj_y must be 1D arrays of equal length")

  time_lags = np.asarray(time_lags, dtype=np.int64)
  n_points = traj_x.shape[0]
  valid_lags = time_lags[(time_lags >= 0) & (time_lags < n_points)]
  valid_lags = np.unique(valid_lags)
  if valid_lags.size < 2:
    return {"box_sizes": np.array([]), "box_counts": np.array([]), "D_box": np.nan}

  return box_counting_dimension(traj_x[valid_lags], traj_y[valid_lags], eps_list=eps_list)


def batch_box_counting_dimension(
  traj_x,
  traj_y=None,
  *,
  time_lags: np.ndarray,
  t_fit_min: float | None = None,
  t_fit_max: float | None = None,
  dt_save: float = 1.0,
  eps_list: np.ndarray | None = None,
  include_per_trajectory: bool = True,
) -> dict:
  """Compute D_box for each trajectory using MSD-like time-lag subsampling."""
  traj_x, traj_y = _unpack_trajectory_inputs(traj_x, traj_y)
  if traj_x.ndim != 2 or traj_y.ndim != 2 or traj_x.shape != traj_y.shape:
    raise ValueError("traj_x and traj_y must have shape (N, T)")

  time_lags = np.asarray(time_lags, dtype=np.int64)
  if t_fit_min is not None or t_fit_max is not None:
    lag_times = time_lags.astype(np.float64) * dt_save
    lag_mask = np.ones(time_lags.size, dtype=bool)
    if t_fit_min is not None:
      lag_mask &= lag_times >= t_fit_min
    if t_fit_max is not None:
      lag_mask &= lag_times <= t_fit_max
    time_lags = time_lags[lag_mask]

  n_particles = traj_x.shape[0]
  d_box_values = np.empty(n_particles, dtype=_ANALYSIS_DTYPE)
  for i in range(n_particles):
    result = box_counting_dimension_at_lags(
      traj_x[i], traj_y[i], time_lags, eps_list=eps_list
    )
    d_box_values[i] = result["D_box"]

  return {
    "D_box": _summarize_metric(d_box_values, include_per_trajectory=include_per_trajectory),
  }


def plot_box_counting_dimension_distribution(
  d_box_values: np.ndarray,
  *,
  bins: int = 100,
):
  """Plot the ensemble distribution of box-counting dimension D_box."""
  import matplotlib.pyplot as plt

  values = np.asarray(d_box_values, dtype=_ANALYSIS_DTYPE)
  finite = values[np.isfinite(values)]

  fig, ax = plt.subplots()
  if finite.size > 0:
    ax.hist(finite, bins=bins, density=True)
  ax.set_xlabel(r"$D_{box}$")
  ax.set_ylabel("density")
  fig.canvas.manager.set_window_title("Box-counting dimension distribution")
  return fig


def observables_to_json(metrics: dict, *, include_per_trajectory: bool) -> dict:
  """Convert observables dictionaries to JSON-serializable Python types."""
  if isinstance(metrics, dict):
    converted = {}
    for key, value in metrics.items():
      if key == "per_trajectory" and not include_per_trajectory:
        continue
      if key.endswith("_per_trajectory") and not include_per_trajectory:
        continue
      converted[key] = observables_to_json(value, include_per_trajectory=include_per_trajectory)
    return converted
  if isinstance(metrics, np.ndarray):
    return metrics.tolist()
  if isinstance(metrics, (np.floating, np.integer)):
    value = metrics.item()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
      return None
    return value
  if isinstance(metrics, float) and (np.isnan(metrics) or np.isinf(metrics)):
    return None
  return metrics


def plot_velocity_cross_correlations(
  lags: np.ndarray,
  c_parallel: np.ndarray,
  c_perp: np.ndarray,
  dt_save: int = 1,
  *,
  loglog: bool = False,
):
  """Plot parallel and perpendicular velocity cross-correlations."""
  import matplotlib.pyplot as plt

  lag_times = np.asarray(lags, dtype=_ANALYSIS_DTYPE) * dt_save
  c_parallel = np.asarray(c_parallel, dtype=_ANALYSIS_DTYPE)
  c_perp = np.asarray(c_perp, dtype=_ANALYSIS_DTYPE)

  fig, ax = plt.subplots()
  if loglog:
    ax.plot(lag_times, np.abs(c_parallel), label="C_parallel")
    ax.plot(lag_times, np.abs(c_perp), label="C_perp")
    ax.set_xscale("log")
    ax.set_yscale("log")
  else:
    ax.plot(lag_times, c_parallel, label="C_parallel")
    ax.plot(lag_times, c_perp, label="C_perp")
  ax.set_xlabel("lag")
  ax.set_ylabel("correlation")
  ax.legend()
  title = "Velocity cross-correlations (log-log)" if loglog else "Velocity cross-correlations"
  fig.canvas.manager.set_window_title(title)
  return fig


def plot_direction_cross_correlations(
  lag_times: np.ndarray,
  c_parallel: np.ndarray,
  c_perp: np.ndarray,
  *,
  loglog: bool = False,
):
  """Plot direction-based parallel and perpendicular cross-correlations."""
  import matplotlib.pyplot as plt

  lag_times = np.asarray(lag_times, dtype=_ANALYSIS_DTYPE)
  c_parallel = np.asarray(c_parallel, dtype=_ANALYSIS_DTYPE)
  c_perp = np.asarray(c_perp, dtype=_ANALYSIS_DTYPE)

  fig, ax = plt.subplots()
  if loglog:
    ax.plot(lag_times, np.abs(c_parallel), label="C_parallel")
    ax.plot(lag_times, np.abs(c_perp), label="C_perp")
    ax.set_xscale("log")
    ax.set_yscale("log")
  else:
    ax.plot(lag_times, c_parallel, label="C_parallel")
    ax.plot(lag_times, c_perp, label="C_perp")
  ax.set_xlabel("lag")
  ax.set_ylabel("correlation")
  ax.legend()
  title = "Direction cross-correlations (log-log)" if loglog else "Direction cross-correlations"
  fig.canvas.manager.set_window_title(title)
  return fig


def plot_direction_autocorrelations(
  lag_times: np.ndarray,
  dx_corr: np.ndarray,
  dy_corr: np.ndarray,
  *,
  loglog: bool = False,
  show_average: bool = True,
  max_lag: int | None = None,
):
  """Plot direction component autocorrelations ⟨dir_i(t) dir_i(t+lag)⟩."""
  import matplotlib.pyplot as plt

  lag_times = np.asarray(lag_times, dtype=_ANALYSIS_DTYPE)
  dx_corr = np.asarray(dx_corr, dtype=_ANALYSIS_DTYPE)
  dy_corr = np.asarray(dy_corr, dtype=_ANALYSIS_DTYPE)
  c_avg = (dx_corr + dy_corr) / 2

  fig, ax = plt.subplots()
  if loglog:
    if show_average:
      ax.plot(lag_times, np.abs(c_avg), label="avg_dxdx_dydy")
    ax.plot(lag_times, np.abs(dx_corr), label="dxdx_corr")
    ax.plot(lag_times, np.abs(dy_corr), label="dydy_corr")
    ax.set_xscale("log")
    ax.set_yscale("log")
  else:
    if show_average:
      ax.plot(lag_times, c_avg, label="avg_dxdx_dydy")
    ax.plot(lag_times, dx_corr, label="dxdx_corr")
    ax.plot(lag_times, dy_corr, label="dydy_corr")
  ax.set_xlabel("lag")
  ax.set_ylabel("correlation")
  if max_lag is not None:
    ax.set_xlim(0, max_lag)
  ax.legend()
  title = "Direction autocorrelations (log-log)" if loglog else "Direction autocorrelations"
  fig.canvas.manager.set_window_title(title)
  return fig


def plot_rg_distribution(
  rg_sq_values: np.ndarray,
  *,
  bins: int = 100,
  log_x: bool = False,
):
  """Plot the ensemble distribution of radius of gyration R_g = sqrt(rg_sq)."""
  return plot_shape_metric_distributions(
    {"rg_sq": rg_sq_values},
    metric_keys=("rg_sq",),
    bins=bins,
    log_x_metrics=("rg_sq",) if log_x else (),
  )


def plot_shape_metric_distributions(
  metric_values: dict[str, np.ndarray],
  *,
  metric_keys: tuple[str, ...] | list[str] | None = None,
  ncol: int = 4,
  bins: int = 100,
  log_x_metrics: tuple[str, ...] | list[str] = (),
):
  """Plot ensemble distributions for selected trajectory shape metrics."""
  import matplotlib.pyplot as plt

  if metric_keys is None:
    metric_keys = tuple(metric_values.keys())
  metric_keys = tuple(metric_keys)
  log_x_metrics = set(log_x_metrics)

  n_metrics = len(metric_keys)
  nrows = (n_metrics + ncol - 1) // ncol
  fig, axes = plt.subplots(nrows, ncol, figsize=(4 * ncol, 3 * nrows))
  axes = np.atleast_1d(axes).ravel()

  for i, key in enumerate(metric_keys):
    values = _metric_values_for_plot(key, metric_values[key])
    finite = values[np.isfinite(values)]
    ax = axes[i]
    if finite.size > 0:
      ax.hist(finite, bins=bins, density=True)
    ax.set_xlabel(_METRIC_PLOT_LABELS.get(key, key))
    ax.set_ylabel("density")
    if key in log_x_metrics:
      ax.set_xscale("log")

  for j in range(n_metrics, len(axes)):
    axes[j].set_visible(False)

  fig.tight_layout()
  fig.canvas.manager.set_window_title("Shape metric distributions")
  return fig
