"""Walk state representations and initial conditions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .directions import COIN_DIM, RIGHT, direction_to_array


@dataclass
class WalkState:
  probabilities: np.ndarray
  grid_size: int
  time: int = 0

  def __post_init__(self) -> None:
    expected = COIN_DIM * self.grid_size ** 2
    if len(self.probabilities) != expected:
      raise ValueError(f"Probability vector size must be 2*grid_size^2, got {len(self.probabilities)}")


def position_to_index(x: int, y: int, grid_size: int) -> int:
  """Return 0-based start index of the 2 coin-state components at site (x, y)."""
  return (x * grid_size + y) * COIN_DIM


def index_to_position(idx: int, grid_size: int) -> Tuple[int, int, Tuple[int, int]]:
  """Convert 1-based flat index to (x, y, direction)."""
  idx_adj = idx - 1
  coin_idx = idx_adj % COIN_DIM
  pos_idx = idx_adj // COIN_DIM
  y = pos_idx % grid_size
  x = pos_idx // grid_size
  direction = RIGHT if coin_idx == 0 else UP
  return x, y, direction


def gaussian_state(
  grid_size: int,
  *,
  center: Optional[Tuple[int, int]] = None,
  width: float = 10.0,
  coin_state: Tuple[int, int] = RIGHT,
) -> WalkState:
  if center is None:
    center = (grid_size // 2, grid_size // 2)

  probabilities = np.zeros(COIN_DIM * grid_size ** 2)
  x0, y0 = center
  coin_vec = direction_to_array(coin_state)

  for x in range(grid_size):
    for y in range(grid_size):
      idx = position_to_index(x, y, grid_size)
      weight = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / width)
      probabilities[idx:idx + COIN_DIM] = weight * coin_vec

  probabilities /= np.linalg.norm(probabilities, ord=1)
  return WalkState(probabilities, grid_size)


def sharp_gaussian_state(
  grid_size: int,
  *,
  center: Optional[Tuple[int, int]] = None,
  coin_state: Tuple[int, int] = RIGHT,
) -> WalkState:
  return gaussian_state(grid_size, center=center, width=1.0, coin_state=coin_state)


def point_state(
  grid_size: int,
  *,
  position: Optional[Tuple[int, int]] = None,
  coin_state: Tuple[int, int] = RIGHT,
) -> WalkState:
  if position is None:
    position = (grid_size // 2, grid_size // 2)

  probabilities = np.zeros(COIN_DIM * grid_size ** 2)
  x, y = position
  idx = position_to_index(x, y, grid_size)
  probabilities[idx:idx + COIN_DIM] = direction_to_array(coin_state)

  return WalkState(probabilities, grid_size)


def get_probability_distribution(state: WalkState) -> np.ndarray:
  d = state.grid_size
  prob_dist = np.zeros((d, d))

  for x in range(d):
    for y in range(d):
      idx = position_to_index(x, y, d)
      coin_vec = state.probabilities[idx:idx + COIN_DIM]
      prob_dist[x, y] = np.linalg.norm(coin_vec, ord=1)

  return prob_dist
