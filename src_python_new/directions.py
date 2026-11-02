"""Cardinal direction constants and helpers for the 2-component coin model."""

from __future__ import annotations

import random
from typing import Sequence, Tuple

import numpy as np

RIGHT: Tuple[int, int] = (1, 0)
LEFT: Tuple[int, int] = (-1, 0)
UP: Tuple[int, int] = (0, 1)
DOWN: Tuple[int, int] = (0, -1)

CARDINAL_DIRECTIONS: Tuple[Tuple[int, int], ...] = (RIGHT, LEFT, UP, DOWN)
COIN_DIM = 2


def direction_to_array(direction: Tuple[int, int]) -> np.ndarray:
  return np.array(direction, dtype=float)


def array_to_direction(vector: Sequence[float]) -> Tuple[int, int]:
  """Snap a 2-vector to the nearest cardinal direction."""
  v = np.asarray(vector, dtype=float)
  best = min(CARDINAL_DIRECTIONS, key=lambda d: np.linalg.norm(v - np.array(d)))
  return best


def random_direction() -> Tuple[int, int]:
  return random.choice(CARDINAL_DIRECTIONS)
