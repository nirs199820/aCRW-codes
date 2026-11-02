"""Coin operators for 2D random walks (2x2 coin matrices)."""

from __future__ import annotations

import random
from abc import ABC
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from .directions import (
  CARDINAL_DIRECTIONS,
  COIN_DIM,
  direction_to_array,
)
from .states import WalkState, position_to_index


PERSISTENCE_MATRIX = np.eye(2)
CHIRAL_MATRIX = np.array([
  [0.0, -1.0],
  [1.0, 0.0],
])

REFLECTION_MATRIX_X = np.array([
  [-1.0, 0.0],
  [0.0, 1.0],
])
REFLECTION_MATRIX_Y = np.array([
  [1.0, 0.0],
  [0.0, -1.0],
])
REFLECTION_MATRIX = np.array([
  [-1.0, 0.0],
  [0.0, -1.0],
])


class CoinOperator(ABC):
  pass


@dataclass
class PersistenceCoin(CoinOperator):
  p_q: float


@dataclass
class ChiralCoin(CoinOperator):
  p_xi: float


@dataclass
class GeneralCoin(CoinOperator):
  p_q: float
  p_xi: float

  def __post_init__(self) -> None:
    if abs(self.p_q) + abs(self.p_xi) > 1.0:
      raise ValueError("|p_q| + |p_xi| must be <= 1")

  @property
  def q_0(self) -> float:
    return 1.0 - abs(self.p_q) - abs(self.p_xi)


@dataclass
class CustomCoin(CoinOperator):
  matrix: np.ndarray

  def __post_init__(self) -> None:
    if self.matrix.shape != (COIN_DIM, COIN_DIM):
      raise ValueError("Coin matrix must be 2x2")
    if not is_stochastic(self.matrix):
      raise ValueError("Coin matrix must be stochastic (rows sum to 1)")


def is_stochastic(matrix: np.ndarray, tol: float = 1e-10) -> bool:
  row_sums = matrix.sum(axis=1)
  return bool(np.allclose(row_sums, 1.0, atol=tol))


def persistence_coin(p_q: float) -> PersistenceCoin:
  return PersistenceCoin(p_q)


def chiral_coin(p_xi: float) -> ChiralCoin:
  return ChiralCoin(p_xi)


def general_coin(p_q: float, p_xi: float) -> GeneralCoin:
  return GeneralCoin(p_q, p_xi)


def identity_coin() -> CustomCoin:
  return CustomCoin(np.eye(COIN_DIM))


def drift_coin(px: float, py: float) -> CustomCoin:
  """Biased coin with drift parameters px (right/left) and py (up/down)."""
  matrix = np.ones((COIN_DIM, COIN_DIM)) / COIN_DIM
  if px > 0:
    matrix[:, 0] += px / 2.0
    matrix[:, 1] -= px / 2.0
  elif px < 0:
    matrix[:, 1] += (-px) / 2.0
    matrix[:, 0] -= (-px) / 2.0
  if py > 0:
    matrix[0, :] += py / 2.0
    matrix[1, :] -= py / 2.0
  elif py < 0:
    matrix[1, :] += (-py) / 2.0
    matrix[0, :] -= (-py) / 2.0
  matrix = np.clip(matrix, 0.0, None)
  matrix /= matrix.sum(axis=1, keepdims=True)
  return CustomCoin(matrix)


def get_coin_matrix(coin: CoinOperator) -> np.ndarray:
  if isinstance(coin, PersistenceCoin):
    return PERSISTENCE_MATRIX * coin.p_q

  if isinstance(coin, ChiralCoin):
    return CHIRAL_MATRIX * coin.p_xi

  if isinstance(coin, GeneralCoin):
    return coin.p_q * PERSISTENCE_MATRIX + coin.p_xi * CHIRAL_MATRIX

  if isinstance(coin, CustomCoin):
    return coin.matrix

  raise TypeError(f"Unknown coin type: {type(coin)}")


def apply_coin_operator(state: WalkState, coin: CoinOperator) -> None:
  d = state.grid_size
  coin_matrix = get_coin_matrix(coin)

  for x in range(d):
    for y in range(d):
      pos_idx = position_to_index(x, y, d)
      probs = state.probabilities[pos_idx:pos_idx + COIN_DIM]
      state.probabilities[pos_idx:pos_idx + COIN_DIM] = coin_matrix @ probs


def direction_probabilities_from_coin_output(p0: float, p1: float) -> np.ndarray:
  """Map coin output (P0, P1) to probabilities over RIGHT, LEFT, UP, DOWN."""
  uniform = (1.0 - abs(p0) - abs(p1)) / 4.0
  probs = np.array([
    max(p0, 0.0) + uniform,
    max(-p0, 0.0) + uniform,
    max(p1, 0.0) + uniform,
    max(-p1, 0.0) + uniform,
  ], dtype=float)
  total = probs.sum()
  if total <= 0.0:
    return np.ones(4) / 4.0
  return probs / total


DIRECTION_TABLE = np.array(CARDINAL_DIRECTIONS, dtype=int)


def direction_probabilities_from_coin_output_batch(outputs: np.ndarray) -> np.ndarray:
  """Map batched coin outputs (N, 2) to direction probabilities (N, 4)."""
  p0 = outputs[:, 0]
  p1 = outputs[:, 1]
  uniform = (1.0 - np.abs(p0) - np.abs(p1)) / 4.0
  probs = np.stack([
    np.maximum(p0, 0.0) + uniform,
    np.maximum(-p0, 0.0) + uniform,
    np.maximum(p1, 0.0) + uniform,
    np.maximum(-p1, 0.0) + uniform,
  ], axis=1)
  totals = probs.sum(axis=1, keepdims=True)
  return np.where(totals <= 0.0, 0.25, probs / totals)


def sample_coin_direction_batch_from_coin(
  coin: np.ndarray,
  dir_idx: np.ndarray,
  rng: np.random.Generator,
) -> np.ndarray:
  """Apply a single coin and sample next direction indices for all particles."""
  dirs = DIRECTION_TABLE[dir_idx]
  outputs = dirs @ coin.T
  probs = direction_probabilities_from_coin_output_batch(outputs)
  u = rng.random(dir_idx.shape[0])
  cumulative = np.cumsum(probs, axis=1)
  return (u[:, None] <= cumulative).argmax(axis=1)


def sample_coin_directions_batch_from_coin_list(
  fixed_coins_list: Sequence[np.ndarray],
  dir_idx_list: np.ndarray,
  rng: np.random.Generator,
) -> np.ndarray:
  """Apply per-particle coins and sample next direction indices for all particles."""
  coins = np.ascontiguousarray(fixed_coins_list)
  dir_idx = np.asarray(dir_idx_list, dtype=np.int64)
  dirs = DIRECTION_TABLE[dir_idx]
  outputs = np.einsum("ni,nji->nj", dirs, coins)
  probs = direction_probabilities_from_coin_output_batch(outputs)
  u = rng.random(dir_idx.shape[0])
  cumulative = np.cumsum(probs, axis=1)
  return (u[:, None] <= cumulative).argmax(axis=1)


def sample_coin_direction_batch(
  coin_open: np.ndarray,
  coin_blocked: np.ndarray,
  on_open: np.ndarray,
  dir_idx: np.ndarray,
  rng: np.random.Generator,
) -> np.ndarray:
  """Apply site-dependent coins and sample next direction indices for all particles."""
  dirs = DIRECTION_TABLE[dir_idx]
  out_open = dirs @ coin_open.T
  out_refl = dirs @ coin_blocked.T
  outputs = np.where(on_open[:, None], out_open, out_refl)
  probs = direction_probabilities_from_coin_output_batch(outputs)
  u = rng.random(dir_idx.shape[0])
  cumulative = np.cumsum(probs, axis=1)
  return (u[:, None] <= cumulative).argmax(axis=1)


def sample_coin_direction(coin: np.ndarray, old_dir: Tuple[int, int]) -> Tuple[int, int]:
  """Apply coin to a cardinal direction and sample the next direction."""
  output = coin @ direction_to_array(old_dir)
  probs = direction_probabilities_from_coin_output(float(output[0]), float(output[1]))
  r = random.random()
  cumulative = 0.0
  for direction, probability in zip(CARDINAL_DIRECTIONS, probs):
    cumulative += probability
    if r <= cumulative:
      return direction
  return CARDINAL_DIRECTIONS[-1]


def get_coin_probabilities(coin: CoinOperator, input_direction: Tuple[int, int]) -> np.ndarray:
  matrix = get_coin_matrix(coin)
  output = matrix @ direction_to_array(input_direction)
  return direction_probabilities_from_coin_output(float(output[0]), float(output[1]))


def create_reflection_coin() -> CustomCoin:
  return CustomCoin(REFLECTION_MATRIX.copy())
