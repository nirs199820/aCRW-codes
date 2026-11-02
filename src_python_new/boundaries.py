"""Boundary conditions and step operators."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .directions import COIN_DIM
from .states import WalkState, position_to_index


class BoundaryCondition(ABC):
  pass


class PeriodicBoundary(BoundaryCondition):
  pass


class ReflectingBoundary(BoundaryCondition):
  pass


class AbsorbingBoundary(BoundaryCondition):
  pass


class BoxBoundary(BoundaryCondition):
  pass


@dataclass
class CustomBoundary(BoundaryCondition):
  boundary_function: Callable[[int, int, int], np.ndarray]


def get_step_operator(grid_size: int, boundary: BoundaryCondition) -> np.ndarray:
  if isinstance(boundary, (PeriodicBoundary, BoxBoundary, ReflectingBoundary, AbsorbingBoundary)):
    return _periodic_step_operator(grid_size)
  raise TypeError(f"Unsupported boundary type for step operator: {type(boundary)}")


def _periodic_step_operator(d: int) -> np.ndarray:
  """Build step operator via signed transport on each standard basis vector."""
  dim = COIN_DIM * d * d
  step_op = np.zeros((dim, dim))

  for col in range(dim):
    state = np.zeros(dim)
    state[col] = 1.0
    temp = _signed_transport_vector(state, d, periodic=True)
    step_op[:, col] = temp

  return step_op


def _signed_transport_vector(
  probabilities: np.ndarray,
  grid_size: int,
  *,
  periodic: bool,
) -> np.ndarray:
  new_probs = np.zeros_like(probabilities)

  for x in range(grid_size):
    for y in range(grid_size):
      idx = position_to_index(x, y, grid_size)
      v0, v1 = probabilities[idx:idx + COIN_DIM]
      v0_pos, v0_neg = max(v0, 0.0), min(v0, 0.0)
      v1_pos, v1_neg = max(v1, 0.0), min(v1, 0.0)

      for nx, ny, delta in (
        (_neighbor(x + 1, grid_size, periodic), y, [v0_pos, 0.0]),
        (_neighbor(x - 1, grid_size, periodic), y, [v0_neg, 0.0]),
        (x, _neighbor(y + 1, grid_size, periodic), [0.0, v1_pos]),
        (x, _neighbor(y - 1, grid_size, periodic), [0.0, v1_neg]),
      ):
        if nx is None or ny is None:
          continue
        ni = position_to_index(nx, ny, grid_size)
        new_probs[ni:ni + COIN_DIM] += delta

  return new_probs


def _neighbor(coord: int, grid_size: int, periodic: bool) -> int | None:
  if periodic:
    return coord % grid_size
  if coord < 0 or coord >= grid_size:
    return None
  return coord


def apply_step_operator(
  state: WalkState,
  boundary: BoundaryCondition,
) -> None:
  periodic = isinstance(boundary, (PeriodicBoundary, BoxBoundary, ReflectingBoundary, AbsorbingBoundary))
  state.probabilities = _signed_transport_vector(
    state.probabilities,
    state.grid_size,
    periodic=periodic,
  )


def is_boundary_site(x: int, y: int, grid_size: int) -> bool:
  return x == 0 or x == grid_size - 1 or y == 0 or y == grid_size - 1


def is_corner(x: int, y: int, grid_size: int) -> bool:
  return (x == 0 or x == grid_size - 1) and (y == 0 or y == grid_size - 1)


def is_vertical_boundary(x: int, grid_size: int) -> bool:
  return x == 0 or x == grid_size - 1


def is_horizontal_boundary(y: int, grid_size: int) -> bool:
  return y == 0 or y == grid_size - 1


def get_boundary_type(x: int, y: int, grid_size: int) -> str:
  at_left = x == 0
  at_right = x == grid_size - 1
  at_bottom = y == 0
  at_top = y == grid_size - 1

  if at_left and at_top:
    return "corner_tl"
  if at_right and at_top:
    return "corner_tr"
  if at_left and at_bottom:
    return "corner_bl"
  if at_right and at_bottom:
    return "corner_br"
  if at_left:
    return "left"
  if at_right:
    return "right"
  if at_bottom:
    return "bottom"
  if at_top:
    return "top"
  return "interior"


def create_perfect_reflection_coin() -> np.ndarray:
  return np.array([
    [-1.0, 0.0],
    [0.0, -1.0],
  ], dtype=complex)


def periodic_boundary() -> PeriodicBoundary:
  return PeriodicBoundary()


def reflecting_boundary() -> ReflectingBoundary:
  return ReflectingBoundary()


def absorbing_boundary() -> AbsorbingBoundary:
  return AbsorbingBoundary()


def box_boundary() -> BoxBoundary:
  return BoxBoundary()
