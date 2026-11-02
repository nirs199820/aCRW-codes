"""Boundary-aware coin operators."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .boundaries import (
  BoxBoundary,
  BoundaryCondition,
  CustomBoundary,
  ReflectingBoundary,
  get_boundary_type,
)
from .coins import REFLECTION_MATRIX, CoinOperator, get_coin_matrix
from .directions import COIN_DIM
from .states import WalkState, position_to_index


def apply_coin_operator_with_boundaries(
  state: WalkState,
  base_coin: CoinOperator,
  boundary: BoundaryCondition,
) -> None:
  d = state.grid_size

  for x in range(d):
    for y in range(d):
      if isinstance(boundary, (BoxBoundary, ReflectingBoundary)):
        coin_matrix = get_boundary_coin_matrix(x, y, d, base_coin, boundary)
      elif isinstance(boundary, CustomBoundary):
        coin_matrix = boundary.boundary_function(x, y, d)
      else:
        coin_matrix = get_coin_matrix(base_coin)

      pos_idx = position_to_index(x, y, d)
      probs = state.probabilities[pos_idx:pos_idx + COIN_DIM]
      state.probabilities[pos_idx:pos_idx + COIN_DIM] = coin_matrix @ probs


def apply_coin_operator_optimized(
  state: WalkState,
  coin_lookup: Dict[str, np.ndarray],
) -> None:
  d = state.grid_size

  for x in range(d):
    for y in range(d):
      boundary_type = get_boundary_type(x, y, d)
      coin_matrix = coin_lookup[boundary_type]

      pos_idx = position_to_index(x, y, d)
      probs = state.probabilities[pos_idx:pos_idx + COIN_DIM]
      state.probabilities[pos_idx:pos_idx + COIN_DIM] = coin_matrix @ probs


def get_boundary_coin_matrix(
  x: int,
  y: int,
  grid_size: int,
  base_coin: CoinOperator,
  boundary: BoxBoundary | ReflectingBoundary,
) -> np.ndarray:
  if isinstance(boundary, ReflectingBoundary):
    boundary = BoxBoundary()

  at_left_boundary = x == 0
  at_right_boundary = x == grid_size - 1
  at_bottom_boundary = y == 0
  at_top_boundary = y == grid_size - 1

  if not (at_left_boundary or at_right_boundary or at_bottom_boundary or at_top_boundary):
    return get_coin_matrix(base_coin)

  return REFLECTION_MATRIX.copy()


def create_boundary_coin_lookup(
  grid_size: int,
  base_coin: CoinOperator,
) -> Dict[str, np.ndarray]:
  lookup: Dict[str, np.ndarray] = {}

  base_matrix = get_coin_matrix(base_coin)
  boundary = BoxBoundary()

  lookup["interior"] = base_matrix

  sample_positions = {
    "left": (0, grid_size // 2),
    "right": (grid_size - 1, grid_size // 2),
    "bottom": (grid_size // 2, 0),
    "top": (grid_size // 2, grid_size - 1),
    "corner_tl": (0, grid_size - 1),
    "corner_tr": (grid_size - 1, grid_size - 1),
    "corner_bl": (0, 0),
    "corner_br": (grid_size - 1, 0),
  }

  for boundary_type, (sx, sy) in sample_positions.items():
    lookup[boundary_type] = get_boundary_coin_matrix(sx, sy, grid_size, base_coin, boundary)

  return lookup
