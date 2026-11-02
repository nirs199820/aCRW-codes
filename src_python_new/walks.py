"""Grid-free stochastic walk simulations."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .coins import (
  CHIRAL_MATRIX,
  DIRECTION_TABLE,
  PERSISTENCE_MATRIX,
  sample_coin_direction_batch_from_coin,
  sample_coin_directions_batch_from_coin_list,
)

_TRAJ_DTYPE = np.int32


def _standard_coin_matrices_from_waiting_times(waiting_times: np.ndarray, chiral_factor: float = 1.0) -> np.ndarray:
  wt = np.maximum(np.asarray(waiting_times, dtype=float), 1.0)
  p_xi = np.minimum(chiral_factor / wt, 1.0)
  p_q = 1.0 - p_xi
  return p_q[:, None, None] * PERSISTENCE_MATRIX + p_xi[:, None, None] * CHIRAL_MATRIX

def simulate_free_diffusion_ensemble_fixed_single_coin_batched(
  coin: np.ndarray,
  save_times: List[int],
  *,
  N_particles: int,
  x0: int,
  y0: int,
  max_steps: int,
  return_directions: bool = False,
  step_size_sampler = None,
  waiting_time_sampler = None,
  print_debug = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  n_save = len(save_times)
  rng = np.random.default_rng()

  x = np.full(N_particles, x0, dtype=_TRAJ_DTYPE)
  y = np.full(N_particles, y0, dtype=_TRAJ_DTYPE)
  dir_idx = rng.integers(0, 4, size=N_particles)

  traj_x = np.zeros((N_particles, n_save), dtype=_TRAJ_DTYPE)
  traj_y = np.zeros((N_particles, n_save), dtype=_TRAJ_DTYPE)
  traj_x[:, 0] = x0
  traj_y[:, 0] = y0
  traj_directions_x = None
  traj_directions_y = None
  if return_directions:
    traj_directions_x = np.zeros((N_particles, n_save), dtype=_TRAJ_DTYPE)
    traj_directions_y = np.zeros((N_particles, n_save), dtype=_TRAJ_DTYPE)

  save_ptr = 1
  draw_direction_at_step = np.ones(N_particles, dtype=_TRAJ_DTYPE)
  for step in range(1, max_steps + 1):

    if step_size_sampler is not None:
      step_size = step_size_sampler() #length N particle array
    else:
      step_size = 1

    if waiting_time_sampler is not None:
      waiting_time = waiting_time_sampler()
    else:
      waiting_time = 1

    moved = step >= draw_direction_at_step
    dir_idx_prev = dir_idx
    dir_idx = sample_coin_direction_batch_from_coin(coin, dir_idx, rng)
    dir_idx = dir_idx_prev * (1 - moved) + dir_idx * moved
    dx = DIRECTION_TABLE[dir_idx, 0] * step_size
    dy = DIRECTION_TABLE[dir_idx, 1] * step_size
    x = x + dx
    y = y + dy
    draw_direction_at_step = np.where(step >= draw_direction_at_step, draw_direction_at_step + waiting_time, draw_direction_at_step)
    
    if print_debug:
      print(f"X={x}, Y={y}\n")
      print(f"dx={dx}, dy={dy}, step_size={step_size}, waiting_time={waiting_time}")
    
    if save_ptr < n_save and step == save_times[save_ptr]:
      traj_x[:, save_ptr] = x
      traj_y[:, save_ptr] = y
      if return_directions:
        traj_directions_x[:, save_ptr] = dx
        traj_directions_y[:, save_ptr] = dy
      save_ptr += 1

  if return_directions:
    return traj_x, traj_y, traj_directions_x, traj_directions_y
  return traj_x, traj_y


def simulate_free_diffusion_ensemble_chiral_runs_batched(
  coin: np.ndarray,
  save_times: List[int],
  *,
  N_particles: int,
  x0: int,
  y0: int,
  max_steps: int,
  waiting_time_sampler,
  chiral_factor: float = 1.0,
  return_directions: bool = False,
  print_debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Free diffusion with alternating chiral runs and inter-run coin steps.

  Each run lasts ``waiting_time`` steps (sampled per particle at run start). During
  a run the particle redraws its direction every step using a standard coin with
  ``p_xi = 1 / waiting_time`` and ``p_q = 1 - p_xi``. Between runs the passed
  ``coin`` is used for one step.
  """
  n_save = len(save_times)
  rng = np.random.default_rng()

  x = np.full(N_particles, x0, dtype=_TRAJ_DTYPE)
  y = np.full(N_particles, y0, dtype=_TRAJ_DTYPE)
  dir_idx = rng.integers(0, 4, size=N_particles)

  traj_x = np.zeros((N_particles, n_save), dtype=_TRAJ_DTYPE)
  traj_y = np.zeros((N_particles, n_save), dtype=_TRAJ_DTYPE)
  traj_x[:, 0] = x0
  traj_y[:, 0] = y0
  traj_directions_x = None
  traj_directions_y = None
  if return_directions:
    traj_directions_x = np.zeros((N_particles, n_save), dtype=_TRAJ_DTYPE)
    traj_directions_y = np.zeros((N_particles, n_save), dtype=_TRAJ_DTYPE)

  save_ptr = 1
  run_waiting_time = waiting_time_sampler()
  run_coins = _standard_coin_matrices_from_waiting_times(run_waiting_time, chiral_factor)
  next_between_step = run_waiting_time + 1
  active_coins = run_coins.copy()

  for step in range(1, max_steps + 1):
    in_between = step >= next_between_step
    active_coins[:] = run_coins
    active_coins[in_between] = coin

    if np.any(in_between):
      new_waiting_time = waiting_time_sampler()
      run_waiting_time = np.where(in_between, new_waiting_time, run_waiting_time)
      run_coins[in_between] = _standard_coin_matrices_from_waiting_times(run_waiting_time, chiral_factor)[in_between]
      next_between_step = np.where(
        in_between,
        step + new_waiting_time + 1,
        next_between_step,
      )

    dir_idx = sample_coin_directions_batch_from_coin_list(active_coins, dir_idx, rng)
    dx = DIRECTION_TABLE[dir_idx, 0]
    dy = DIRECTION_TABLE[dir_idx, 1]
    x = x + dx
    y = y + dy

    if print_debug:
      print(f"step={step}, in_between={in_between}")
      print(f"X={x}, Y={y}")
      print(f"dx={dx}, dy={dy}, run_waiting_time={run_waiting_time}")

    if save_ptr < n_save and step == save_times[save_ptr]:
      traj_x[:, save_ptr] = x
      traj_y[:, save_ptr] = y
      if return_directions:
        traj_directions_x[:, save_ptr] = dx
        traj_directions_y[:, save_ptr] = dy
      save_ptr += 1

  if return_directions:
    return traj_x, traj_y, traj_directions_x, traj_directions_y
  return traj_x, traj_y


def simulate_free_diffusion_ensemble_time_dependent_coins_batched(
  time_dependent_coins: List[np.ndarray],
  save_times: List[int],
  *,
  N_particles: int,
  x0: int,
  y0: int,
  max_steps: int,
  return_directions: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  n_save = len(save_times)
  rng = np.random.default_rng()

  x = np.full(N_particles, x0, dtype=_TRAJ_DTYPE)
  y = np.full(N_particles, y0, dtype=_TRAJ_DTYPE)
  dir_idx = rng.integers(0, 4, size=N_particles)

  traj_x = np.zeros((N_particles, n_save), dtype=_TRAJ_DTYPE)
  traj_y = np.zeros((N_particles, n_save), dtype=_TRAJ_DTYPE)
  traj_x[:, 0] = x0
  traj_y[:, 0] = y0
  traj_directions_x = None
  traj_directions_y = None
  if return_directions:
    traj_directions_x = np.zeros((N_particles, n_save), dtype=_TRAJ_DTYPE)
    traj_directions_y = np.zeros((N_particles, n_save), dtype=_TRAJ_DTYPE)

  save_ptr = 1
  for step in range(1, max_steps + 1):
    coin = time_dependent_coins[step - 1]
    dir_idx = sample_coin_direction_batch_from_coin(coin, dir_idx, rng)

    dx = DIRECTION_TABLE[dir_idx, 0]
    dy = DIRECTION_TABLE[dir_idx, 1]
    x = x + dx
    y = y + dy

    if save_ptr < n_save and step == save_times[save_ptr]:
      traj_x[:, save_ptr] = x
      traj_y[:, save_ptr] = y
      if return_directions:
        traj_directions_x[:, save_ptr] = dx
        traj_directions_y[:, save_ptr] = dy
      save_ptr += 1

  if return_directions:
    return traj_x, traj_y, traj_directions_x, traj_directions_y
  return traj_x, traj_y

def simulate_free_diffusion_ensemble_fixed_coins_list_batched(
  fixed_coins_list: List[np.ndarray],
  save_times: List[int],
  *,
  N_particles: int,
  x0: int,
  y0: int,
  max_steps: int,
  return_directions: bool = False,
  step_size_sampler = None,
  waiting_time_sampler = None,
  print_debug = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  n_save = len(save_times)
  rng = np.random.default_rng()

  x = np.full(N_particles, x0, dtype=_TRAJ_DTYPE)
  y = np.full(N_particles, y0, dtype=_TRAJ_DTYPE)
  dir_idx = rng.integers(0, 4, size=N_particles)

  traj_x = np.zeros((N_particles, n_save), dtype=_TRAJ_DTYPE)
  traj_y = np.zeros((N_particles, n_save), dtype=_TRAJ_DTYPE)
  traj_x[:, 0] = x0
  traj_y[:, 0] = y0
  traj_directions_x = None
  traj_directions_y = None
  if return_directions:
    traj_directions_x = np.zeros((N_particles, n_save), dtype=_TRAJ_DTYPE)
    traj_directions_y = np.zeros((N_particles, n_save), dtype=_TRAJ_DTYPE)

  save_ptr = 1
  draw_direction_at_step = np.ones(N_particles, dtype=_TRAJ_DTYPE)
  for step in range(1, max_steps + 1):

    if step_size_sampler is not None:
      step_size = step_size_sampler() #length N particle array
    else:
      step_size = 1

    if waiting_time_sampler is not None:
      waiting_time = waiting_time_sampler()
    else:
      waiting_time = 1

    moved = step >= draw_direction_at_step
    dir_idx_prev = dir_idx
    dir_idx = sample_coin_directions_batch_from_coin_list(fixed_coins_list, dir_idx, rng)
    dir_idx = dir_idx_prev * (1 - moved) + dir_idx * moved
    dx = DIRECTION_TABLE[dir_idx, 0] * step_size
    dy = DIRECTION_TABLE[dir_idx, 1] * step_size
    x = x + dx
    y = y + dy
    draw_direction_at_step = np.where(step >= draw_direction_at_step, draw_direction_at_step + waiting_time, draw_direction_at_step)
    
    if print_debug:
      print(f"X={x}, Y={y}\n")
      print(f"dx={dx}, dy={dy}, step_size={step_size}, waiting_time={waiting_time}")
    
    if save_ptr < n_save and step == save_times[save_ptr]:
      traj_x[:, save_ptr] = x
      traj_y[:, save_ptr] = y
      if return_directions:
        traj_directions_x[:, save_ptr] = dx
        traj_directions_y[:, save_ptr] = dy
      save_ptr += 1

  if return_directions:
    return traj_x, traj_y, traj_directions_x, traj_directions_y
  return traj_x, traj_y


def simulate_free_diffusion_ensemble_particle_time_coins_batched(
  particle_time_coins: np.ndarray,
  save_times: List[int],
  *,
  N_particles: int,
  x0: int,
  y0: int,
  max_steps: int,
  return_directions: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  expected_shape = (N_particles, max_steps, 2, 2)
  particle_time_coins = np.ascontiguousarray(particle_time_coins)
  if particle_time_coins.shape != expected_shape:
    raise ValueError(
      f"particle_time_coins must have shape {expected_shape}, got {particle_time_coins.shape}"
    )

  n_save = len(save_times)
  rng = np.random.default_rng()

  x = np.full(N_particles, x0, dtype=_TRAJ_DTYPE)
  y = np.full(N_particles, y0, dtype=_TRAJ_DTYPE)
  dir_idx = rng.integers(0, 4, size=N_particles)

  traj_x = np.zeros((N_particles, n_save), dtype=_TRAJ_DTYPE)
  traj_y = np.zeros((N_particles, n_save), dtype=_TRAJ_DTYPE)
  traj_x[:, 0] = x0
  traj_y[:, 0] = y0
  traj_directions_x = None
  traj_directions_y = None
  if return_directions:
    traj_directions_x = np.zeros((N_particles, n_save), dtype=_TRAJ_DTYPE)
    traj_directions_y = np.zeros((N_particles, n_save), dtype=_TRAJ_DTYPE)

  save_ptr = 1
  for step in range(1, max_steps + 1):
    dir_idx = sample_coin_directions_batch_from_coin_list(
      particle_time_coins[:, step - 1], dir_idx, rng
    )

    dx = DIRECTION_TABLE[dir_idx, 0]
    dy = DIRECTION_TABLE[dir_idx, 1]
    x = x + dx
    y = y + dy

    if save_ptr < n_save and step == save_times[save_ptr]:
      traj_x[:, save_ptr] = x
      traj_y[:, save_ptr] = y
      if return_directions:
        traj_directions_x[:, save_ptr] = dx
        traj_directions_y[:, save_ptr] = dy
      save_ptr += 1

  if return_directions:
    return traj_x, traj_y, traj_directions_x, traj_directions_y
  return traj_x, traj_y
