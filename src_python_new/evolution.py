"""Evolution engine and simulation routines."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .boundaries import (
  AbsorbingBoundary,
  BoxBoundary,
  BoundaryCondition,
  CustomBoundary,
  PeriodicBoundary,
  ReflectingBoundary,
  apply_step_operator,
  get_step_operator,
)
from .boundary_coins import (
  apply_coin_operator_optimized,
  apply_coin_operator_with_boundaries,
  create_boundary_coin_lookup,
)
from .coins import (
  ChiralCoin,
  CoinOperator,
  GeneralCoin,
  PersistenceCoin,
  apply_coin_operator,
  drift_coin,
  general_coin,
  get_coin_matrix,
  persistence_coin,
  sample_coin_direction,
)
from .directions import COIN_DIM


@dataclass
class Evolution:
  grid_size: int
  coin: CoinOperator
  boundary: BoundaryCondition
  step_operator: np.ndarray
  coin_lookup: Optional[Dict[str, np.ndarray]] = None

  def __init__(self, grid_size: int, coin: CoinOperator, boundary: BoundaryCondition):
    self.grid_size = grid_size
    self.coin = coin
    self.boundary = boundary
    self.step_operator = get_step_operator(grid_size, boundary)
    self.coin_lookup = None
    if isinstance(boundary, (BoxBoundary, ReflectingBoundary)):
      self.coin_lookup = create_boundary_coin_lookup(grid_size, coin)


def evolve(state: WalkState, evolution: Evolution, steps: int = 1) -> None:
  for _ in range(steps):
    if isinstance(evolution.boundary, (BoxBoundary, ReflectingBoundary)):
      evolve_reflecting_boundary(state, evolution)
    elif isinstance(evolution.boundary, PeriodicBoundary):
      evolve_periodic(state, evolution)
    elif isinstance(evolution.boundary, AbsorbingBoundary):
      evolve_absorbing(state, evolution)
    elif isinstance(evolution.boundary, CustomBoundary):
      evolve_custom(state, evolution)
    else:
      evolve_standard(state, evolution)


def evolve_reflecting_boundary(state: WalkState, evolution: Evolution) -> None:
  if evolution.coin_lookup is not None:
    apply_coin_operator_optimized(state, evolution.coin_lookup)
  else:
    apply_coin_operator_with_boundaries(state, evolution.coin, evolution.boundary)

  apply_step_operator(state, evolution.boundary)
  state.time += 1


def evolve_periodic(state: WalkState, evolution: Evolution) -> None:
  apply_coin_operator(state, evolution.coin)
  apply_step_operator(state, evolution.boundary)
  state.time += 1


def evolve_absorbing(state: WalkState, evolution: Evolution) -> None:
  """Stub: absorbing boundary uses the same step operator as periodic in Julia."""
  evolve_periodic(state, evolution)


def evolve_custom(state: WalkState, evolution: Evolution) -> None:
  apply_coin_operator_with_boundaries(state, evolution.coin, evolution.boundary)
  apply_step_operator(state, evolution.boundary)
  state.time += 1


def evolve_standard(state: WalkState, evolution: Evolution) -> None:
  apply_coin_operator(state, evolution.coin)
  apply_step_operator(state, evolution.boundary)
  state.time += 1


def create_periodic_evolution(grid_size: int, coin: CoinOperator) -> Evolution:
  return Evolution(grid_size, coin, PeriodicBoundary())


def create_box_evolution(grid_size: int, coin: CoinOperator) -> Evolution:
  return Evolution(grid_size, coin, BoxBoundary())


def create_reflecting_evolution(grid_size: int, coin: CoinOperator) -> Evolution:
  return Evolution(grid_size, coin, ReflectingBoundary())


def create_absorbing_evolution(grid_size: int, coin: CoinOperator) -> Evolution:
  return Evolution(grid_size, coin, AbsorbingBoundary())


def create_custom_evolution(
  grid_size: int,
  coin: CoinOperator,
  custom_boundary_func: Callable[[int, int, int], np.ndarray],
) -> Evolution:
  return Evolution(grid_size, coin, CustomBoundary(custom_boundary_func))


def simulate_walk(
  initial_state: WalkState,
  evolution: Evolution,
  max_time: int,
  *,
  save_trajectory: bool = False,
  save_interval: int = 1,
) -> Tuple[WalkState, List[WalkState]]:
  state = WalkState(initial_state.probabilities.copy(), initial_state.grid_size)

  trajectory: List[WalkState] = []
  if save_trajectory:
    trajectory.append(WalkState(state.probabilities.copy(), state.grid_size))

  for t in range(1, max_time + 1):
    evolve(state, evolution)

    if save_trajectory and t % save_interval == 0:
      saved_state = WalkState(state.probabilities.copy(), state.grid_size)
      saved_state.time = state.time
      trajectory.append(saved_state)

  return state, trajectory


def simulate_walk_batch(
  initial_states: List[WalkState],
  evolution: Evolution,
  max_time: int,
) -> List[WalkState]:
  final_states: List[WalkState] = []
  for initial_state in initial_states:
    final_state, _ = simulate_walk(initial_state, evolution, max_time)
    final_states.append(final_state)
  return final_states


def simulate_walk_with_regions(
  initial_state: WalkState,
  coin: CoinOperator,
  max_time: int,
  regions: Dict[str, Callable[[int, int], bool]],
  *,
  save_trajectory: bool = False,
  save_interval: int = 1,
) -> Tuple[WalkState, List[WalkState]]:
  state = WalkState(initial_state.probabilities.copy(), initial_state.grid_size)

  trajectory: List[WalkState] = []
  if save_trajectory:
    trajectory.append(WalkState(state.probabilities.copy(), state.grid_size))

  for t in range(1, max_time + 1):
    evolve_with_topology(state, coin, state.grid_size, regions)

    if save_trajectory and t % save_interval == 0:
      saved_state = WalkState(state.probabilities.copy(), state.grid_size)
      saved_state.time = state.time
      trajectory.append(saved_state)

  return state, trajectory


def evolve_with_topology(
  state: WalkState,
  coin: CoinOperator,
  grid_size: int,
  coin_regions: Dict[str, Callable[[int, int], bool]],
) -> None:
  apply_step_operator(state, PeriodicBoundary())

  applied_regions = np.zeros((grid_size, grid_size), dtype=bool)

  for region_name, condition_func in coin_regions.items():
    region_coin = get_region_coin(region_name, coin)
    apply_regional_coin_operator(state, region_coin, condition_func, applied_regions)

  apply_default_coin_operator(state, coin, applied_regions)
  state.time += 1


def apply_regional_coin_operator(
  state: WalkState,
  coin: CoinOperator,
  condition_func: Callable[[int, int], bool],
  applied_regions: np.ndarray,
) -> None:
  d = state.grid_size
  coin_matrix = get_coin_matrix(coin)

  for x in range(d):
    for y in range(d):
      if condition_func(x, y):
        pos_idx = position_to_index(x, y, d)
        probs = state.probabilities[pos_idx:pos_idx + COIN_DIM]
        state.probabilities[pos_idx:pos_idx + COIN_DIM] = coin_matrix @ probs
        applied_regions[x, y] = True


def apply_default_coin_operator(
  state: WalkState,
  coin: CoinOperator,
  applied_regions: np.ndarray,
) -> None:
  d = state.grid_size
  coin_matrix = get_coin_matrix(coin)

  for x in range(d):
    for y in range(d):
      if not applied_regions[x, y]:
        pos_idx = position_to_index(x, y, d)
        probs = state.probabilities[pos_idx:pos_idx + COIN_DIM]
        state.probabilities[pos_idx:pos_idx + COIN_DIM] = coin_matrix @ probs


def _general_coin_params(coin: CoinOperator) -> Tuple[float, float]:
  if isinstance(coin, GeneralCoin):
    return coin.p_q, coin.p_xi
  if isinstance(coin, PersistenceCoin):
    return coin.p_q, 0.0
  if isinstance(coin, ChiralCoin):
    return 0.0, coin.p_xi
  return 0.0, 0.0


def get_region_coin(region_name: str, default_coin: CoinOperator) -> CoinOperator:
  p_q, p_xi = _general_coin_params(default_coin)
  if region_name == "left_half":
    return general_coin(p_q, -p_xi)
  if region_name == "right_half":
    return general_coin(p_q, p_xi * 0.8)
  if region_name == "center_line":
    return drift_coin(0.1, 0.0)
  if region_name == "absorbing":
    return persistence_coin(0.0)
  if region_name == "reflecting":
    return persistence_coin(0.0)
  if region_name == "viral_zone":
    return drift_coin(0.2, 0.1)
  if region_name == "upper_half":
    return drift_coin(0.0, 0.1)
  if region_name == "lower_half":
    return drift_coin(0.0, -0.1)
  return default_coin


def create_topological_regions(grid_size: int) -> Dict[str, Callable[[int, int], bool]]:
  regions: Dict[str, Callable[[int, int], bool]] = {}

  regions["left_half"] = lambda x, y: x < grid_size // 2
  regions["right_half"] = lambda x, y: x >= grid_size // 2
  regions["center_line"] = lambda x, y: abs(x - grid_size // 2) <= 1
  regions["upper_half"] = lambda x, y: y > grid_size // 2
  regions["lower_half"] = lambda x, y: y <= grid_size // 2

  center = grid_size // 2
  regions["center_circle"] = (
    lambda x, y: (x - center) ** 2 + (y - center) ** 2 <= (grid_size // 4) ** 2
  )

  def outer_ring(x: int, y: int) -> bool:
    r2 = (x - center) ** 2 + (y - center) ** 2
    return (grid_size // 4) ** 2 < r2 <= (grid_size // 3) ** 2

  regions["outer_ring"] = outer_ring
  return regions


def single_particle_simulation(
  initial_pos: Tuple[int, int],
  coin_state: Tuple[int, int],
  grid_size: int,
  coin: CoinOperator,
  max_time: int,
  num_realizations: int,
) -> np.ndarray:
  distribution = np.zeros((grid_size, grid_size))
  coin_matrix = get_coin_matrix(coin)

  for _ in range(num_realizations):
    x, y = initial_pos
    direction = coin_state

    for _ in range(max_time):
      direction = sample_coin_direction(coin_matrix, direction)
      dx, dy = direction
      x = (x + dx) % grid_size
      y = (y + dy) % grid_size

    distribution[y, x] += 1.0 / num_realizations

  return distribution


def _signed_transport_site(
  temp_probs: np.ndarray,
  state_probs: np.ndarray,
  x: int,
  y: int,
  grid_size: int,
) -> None:
  idx = position_to_index(x, y, grid_size)
  v0, v1 = state_probs[idx:idx + COIN_DIM]
  v0_pos, v0_neg = max(v0, 0.0), min(v0, 0.0)
  v1_pos, v1_neg = max(v1, 0.0), min(v1, 0.0)

  for nx, ny, delta in (
    ((x + 1) % grid_size, y, [v0_pos, 0.0]),
    ((x - 1) % grid_size, y, [v0_neg, 0.0]),
    (x, (y + 1) % grid_size, [0.0, v1_pos]),
    (x, (y - 1) % grid_size, [0.0, v1_neg]),
  ):
    ni = position_to_index(nx, ny, grid_size)
    temp_probs[ni:ni + COIN_DIM] += delta


def simulate_hitting_time(
  initial_state: WalkState,
  target_x: int,
  target_y: int,
  coin_func: Callable[[WalkState], None],
  max_time: int,
  ps_min: float,
) -> Tuple[float, float, float]:
  state = WalkState(initial_state.probabilities.copy(), initial_state.grid_size)
  d = state.grid_size
  target_idx = position_to_index(target_x, target_y, d)
  target_range = slice(target_idx, target_idx + COIN_DIM)

  mht_sum = 0.0
  absorbed = 0.0
  ps = 1.0

  temp_probs = np.zeros_like(state.probabilities)

  for t in range(1, max_time + 1):
    coin_func(state)

    temp_probs.fill(0.0)
    for x in range(d):
      for y in range(d):
        _signed_transport_site(temp_probs, state.probabilities, x, y, d)

    state.probabilities = temp_probs.copy()
    state.time += 1

    p_hit = float(np.linalg.norm(state.probabilities[target_range], ord=1))
    if p_hit > 0:
      mht_sum += t * p_hit
      absorbed += p_hit
      state.probabilities[target_range] = 0.0

    ps = float(np.linalg.norm(state.probabilities, ord=1))
    if ps < ps_min:
      break

  mht = mht_sum / absorbed if absorbed > 0 else 0.0
  return mht, ps, absorbed
