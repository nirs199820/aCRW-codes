"""Master-equation propagation of 4-orientation distributions on an N×N grid."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

BoundaryMode = Literal["periodic", "open", "reflecting", "reflective_infinite_force"]
ReflectionCoinType = Literal[
  "randomized",
  "reflective_infinite_force",
  "reflecting",
  "anti_reflecting",
  "stay",
  "scatter",
  "specular",
]

ORIENTATION_COLS = ("p_right", "p_up", "p_left", "p_down")
ORIENTATION_NAMES = ("right", "up", "left", "down")
ORIENTATION_INDEX = {name: idx for idx, name in enumerate(ORIENTATION_NAMES)}

# 4×4 basis matrices (state order: right, left, up, down)
LEFT_CHIRAL_MATRIX_4 = np.array([
  [0, 0, 0, 1],
  [1, 0, 0, 0],
  [0, 1, 0, 0],
  [0, 0, 1, 0],
], dtype=float)
RIGHT_CHIRAL_MATRIX_4 = np.array([
  [0, 1, 0, 0],
  [0, 0, 1, 0],
  [0, 0, 0, 1],
  [1, 0, 0, 0],
], dtype=float)
PERSISTENCE_MATRIX_4 = np.eye(4, dtype=float)
REVERSE_MATRIX_4 = np.array([
  [0, 0, 1, 0],
  [0, 0, 0, 1],
  [1, 0, 0, 0],
  [0, 1, 0, 0],
], dtype=float)
RANDOM_MATRIX_4 = np.full((4, 4), 0.25, dtype=float)
# Julia src reflection coin: swap right↔left and up↔down at every wall/corner site.
REFLECTION_MATRIX_4 = np.array([
  [0.0, 0.0, 1.0, 0.0],
  [0.0, 0.0, 0.0, 1.0],
  [1.0, 0.0, 0.0, 0.0],
  [0.0, 1.0, 0.0, 0.0],
], dtype=float)
# Blocked heading is redistributed equally over the other three orientations.
SCATTER_MATRIX_4 = (np.ones((4, 4), dtype=float) - np.eye(4, dtype=float)) / 3.0

# Orientation d moves by U_d in real space; pull form uses source r - U_d
SHIFTS = np.array([
  [1, 0],
  [0, 1],
  [-1, 0],
  [0, -1],
], dtype=int)

class ReflectionCoin:
  def __init__(self):
    pass
  
class RandomizedReflectionCoin(ReflectionCoin):
  def __init__(self) -> None:
    super().__init__()
    self.coin = RANDOM_MATRIX_4

class InfiniteForceReflectionCoin(ReflectionCoin):
  def __init__(self) -> None:
    super().__init__()
    self.coin = REFLECTION_MATRIX_4

class ReflectingCoin(ReflectionCoin):
  def __init__(self, coin: np.ndarray) -> None:
    super().__init__()
    self.coin = coin.copy()

class AntiReflectingCoin(ReflectionCoin):
  def __init__(self, coin: np.ndarray) -> None:
    super().__init__()
    self.coin = coin.T.copy()

class StayReflectionCoin(ReflectionCoin):
  """Blocked wall mass keeps its orientation unchanged (identity wall coin)."""

  def __init__(self) -> None:
    super().__init__()
    self.coin = PERSISTENCE_MATRIX_4.copy()

class ScatterReflectionCoin(ReflectionCoin):
  """Blocked heading is split equally into the other three orientations."""

  def __init__(self) -> None:
    super().__init__()
    self.coin = SCATTER_MATRIX_4.copy()

def reflectionCoinPicker(coin: np.ndarray, reflection_coin_type: ReflectionCoinType) -> ReflectionCoin:
  if reflection_coin_type == "randomized":
    return RandomizedReflectionCoin()
  elif reflection_coin_type in ("reflective_infinite_force", "specular"):
    return InfiniteForceReflectionCoin()
  elif reflection_coin_type == "reflecting":
    return ReflectingCoin(coin)
  elif reflection_coin_type == "anti_reflecting":
    return AntiReflectingCoin(coin)
  elif reflection_coin_type == "stay":
    return StayReflectionCoin()
  elif reflection_coin_type == "scatter":
    return ScatterReflectionCoin()
  else:
    raise ValueError(f"Unknown reflection coin type: {reflection_coin_type}")


def make_coin_4x4(p_persistent: float,p_right: float, p_reverse: float, p_left: float) -> np.ndarray:
  if p_left + p_right + p_persistent + p_reverse > 1.0:
    raise ValueError("p_left + p_right + p_persistent + p_reverse must be <= 1")
  p_rand = 1.0 - p_left - p_right - p_persistent - p_reverse
  return (
    p_left * LEFT_CHIRAL_MATRIX_4
    + p_right * RIGHT_CHIRAL_MATRIX_4
    + p_persistent * PERSISTENCE_MATRIX_4
    + p_reverse * REVERSE_MATRIX_4
    + p_rand * RANDOM_MATRIX_4
  )

def make_acp_coin_4x4(p_q: float, p_xi: float) -> np.ndarray:
  """Build a stochastic 4×4 ACP coin from (p_q, p_xi) parameters.

  Positive ``p_xi`` is a counter-clockwise turn (right to up to left to down),
  matching the continuum convention of a positive angular speed.
  """
  if not (np.isfinite(p_q) and np.isfinite(p_xi)):
    raise ValueError("p_q and p_xi must be finite")
  weight_sum = abs(p_q) + abs(p_xi)
  if weight_sum > 1.0:
    raise ValueError("|p_q| + |p_xi| must be <= 1")

  # Use the same grouped sum as the admissibility check. Sequential
  # subtraction gives 1 - 0.9 - 0.1 = -2.78e-17, a negative random weight.
  p_rand = 1.0 - weight_sum
  p_persistent = max(p_q, 0.0)
  p_reverse = max(-p_q, 0.0)
  p_left = max(p_xi, 0.0)
  p_right = max(-p_xi, 0.0)

  return (
    p_rand * RANDOM_MATRIX_4
    + p_left * LEFT_CHIRAL_MATRIX_4
    + p_right * RIGHT_CHIRAL_MATRIX_4
    + p_persistent * PERSISTENCE_MATRIX_4
    + p_reverse * REVERSE_MATRIX_4
  )


def make_coin_field_from_index_map(
  index_map: np.ndarray,
  coins: np.ndarray | Sequence[np.ndarray],
) -> np.ndarray:
  """Build a site-local coin field of shape (N, N, 4, 4) from integer labels.

  ``index_map[x, y]`` selects which coin from ``coins`` applies at that site.
  ``coins`` may be a sequence of (4, 4) matrices or an array of shape (K, 4, 4).
  """
  index_map = np.asarray(index_map, dtype=int)
  if index_map.ndim != 2 or index_map.shape[0] != index_map.shape[1]:
    raise ValueError("index_map must be a square (N, N) array")

  coin_stack = np.asarray(coins, dtype=float)
  if coin_stack.ndim == 2:
    coin_stack = coin_stack[None, ...]
  if coin_stack.ndim != 3 or coin_stack.shape[1:] != (4, 4):
    raise ValueError("coins must have shape (K, 4, 4) or be a sequence of (4, 4)")

  if index_map.min() < 0 or index_map.max() >= coin_stack.shape[0]:
    raise ValueError(
      f"index_map entries must lie in [0, {coin_stack.shape[0] - 1}]"
    )
  return coin_stack[index_map]


def make_half_half_coin_field(
  grid_size: int,
  coin_left: np.ndarray,
  coin_right: np.ndarray,
  *,
  axis: Literal["x", "y"] = "x",
) -> np.ndarray:
  """Split the arena into two halves with different coins.

  Default ``axis="x"``: sites with ``x < N//2`` use ``coin_left``, the rest
  use ``coin_right``. With ``axis="y"``, the split is on the y coordinate.
  """
  if axis not in ("x", "y"):
    raise ValueError("axis must be 'x' or 'y'")
  coin_left = np.asarray(coin_left, dtype=float)
  coin_right = np.asarray(coin_right, dtype=float)
  if coin_left.shape != (4, 4) or coin_right.shape != (4, 4):
    raise ValueError("coin_left and coin_right must have shape (4, 4)")

  index_map = np.zeros((grid_size, grid_size), dtype=int)
  mid = grid_size // 2
  if axis == "x":
    index_map[mid:, :] = 1
  else:
    index_map[:, mid:] = 1
  return make_coin_field_from_index_map(index_map, [coin_left, coin_right])


def normalize_coin_input(coin: np.ndarray, grid_size: int) -> np.ndarray:
  """Accept a global (4, 4) coin or a site-local (N, N, 4, 4) coin field."""
  coin = np.asarray(coin, dtype=float)
  if coin.ndim == 2:
    if coin.shape != (4, 4):
      raise ValueError("global coin must have shape (4, 4)")
    return coin
  if coin.ndim == 4:
    expected = (grid_size, grid_size, 4, 4)
    if coin.shape != expected:
      raise ValueError(f"coin field must have shape {expected}, got {coin.shape}")
    return coin
  raise ValueError("coin must have shape (4, 4) or (N, N, 4, 4)")


def _apply_coin_to_P(P: np.ndarray, coin: np.ndarray) -> np.ndarray:
  """Apply a global or site-local coin to orientation field P."""
  if coin.ndim == 2:
    return np.einsum("ij,xyj->xyi", coin, P)
  return np.einsum("xyij,xyj->xyi", coin, P)


def _wall_coin_field(
  coin: np.ndarray,
  reflection_coin_type: ReflectionCoinType,
) -> np.ndarray:
  """Wall coin(s) used on blocked reflecting hops.

  Global wall types return a single (4, 4) matrix. Site-local ``reflecting`` /
  ``anti_reflecting`` return a field with the same spatial shape as ``coin``.
  ``stay`` keeps blocked orientation unchanged (identity).
  ``scatter`` splits blocked heading equally into the other three directions.
  ``specular`` reverses the blocked heading (right↔left, up↔down), the
  lattice form of the continuum face reflection.
  """
  if reflection_coin_type in (
    "randomized",
    "reflective_infinite_force",
    "specular",
    "stay",
    "scatter",
  ):
    return reflectionCoinPicker(np.eye(4), reflection_coin_type).coin

  if coin.ndim == 2:
    return reflectionCoinPicker(coin, reflection_coin_type).coin

  if reflection_coin_type == "reflecting":
    return coin
  if reflection_coin_type == "anti_reflecting":
    return np.swapaxes(coin, -2, -1)
  raise ValueError(f"Unknown reflection coin type: {reflection_coin_type}")


def _make_grid_index(grid_size: int) -> pd.MultiIndex:
  xs = np.arange(grid_size)
  ys = np.arange(grid_size)
  return pd.MultiIndex.from_product([xs, ys], names=["x", "y"])


def _update_rho(df: pd.DataFrame) -> None:
  df["rho"] = df[list(ORIENTATION_COLS)].sum(axis=1)


@dataclass
class OrientationGrid:
  """Grid-wide orientation-resolved probability field backed by a pandas DataFrame."""

  grid_size: int
  df: pd.DataFrame = field(repr=False)
  time: int = 0
  initial_orientation: Optional[str] = None

  @classmethod
  def from_array(cls, P: np.ndarray, *, time: int = 0) -> OrientationGrid:
    """Create from numpy array with shape (N, N, 4)."""
    if P.ndim != 3 or P.shape[2] != 4:
      raise ValueError("P must have shape (N, N, 4)")

    n = P.shape[0]
    if P.shape[1] != n:
      raise ValueError("Grid must be square")

    flat = P.reshape(-1, 4)
    df = pd.DataFrame(flat, index=_make_grid_index(n), columns=list(ORIENTATION_COLS))
    df["rho"] = flat.sum(axis=1)
    return cls(grid_size=n, df=df, time=time)

  @classmethod
  def zeros(cls, grid_size: int) -> OrientationGrid:
    P = np.zeros((grid_size, grid_size, 4), dtype=float)
    return cls.from_array(P)

  def to_array(self) -> np.ndarray:
    """Return orientation probabilities as (N, N, 4) numpy array."""
    values = self.df[list(ORIENTATION_COLS)].to_numpy(dtype=float)
    return values.reshape(self.grid_size, self.grid_size, 4)

  def rho_array(self) -> np.ndarray:
    """Return marginal spatial density rho as (N, N) array."""
    return snapshot_to_rho_array(self.df, self.grid_size)

  def total_probability(self) -> float:
    return float(self.df["rho"].sum())

  def normalize(self) -> None:
    total = self.total_probability()
    if total > 0.0:
      self.df[list(ORIENTATION_COLS)] /= total
      _update_rho(self.df)

  def copy(self) -> OrientationGrid:
    return OrientationGrid(
      grid_size=self.grid_size,
      df=self.df.copy(),
      time=self.time,
    )

  def snapshot_dataframe(self) -> pd.DataFrame:
    return self.df.copy()


def _resolve_orientation(
  orientation: str,
  rng: Optional[np.random.Generator] = None,
) -> str:
  """Map orientation aliases to a concrete cardinal direction."""
  if orientation == "random":
    generator = rng if rng is not None else np.random.default_rng()
    return str(generator.choice(ORIENTATION_NAMES))
  if orientation == "uniform":
    return "uniform"
  if orientation not in ORIENTATION_INDEX:
    raise ValueError(
      f"orientation must be one of {list(ORIENTATION_NAMES)} or 'random' or 'uniform'"
    )
  return orientation


def point_initial_state(
  grid_size: int,
  position: Tuple[int, int],
  *,
  orientation: str = "right",
  rng: Optional[np.random.Generator] = None,
) -> OrientationGrid:
  """Point source with unit probability at ``position``.

  ``orientation`` options:
  - ``right`` / ``left`` / ``up`` / ``down``: pure state in that direction
  - ``random``: sample one cardinal direction uniformly at random
  - ``uniform``: equal weight 0.25 in all four orientations (unpolarized coin)
  """
  resolved = _resolve_orientation(orientation, rng=rng)

  x0, y0 = position
  if not (0 <= x0 < grid_size and 0 <= y0 < grid_size):
    raise ValueError("position must lie inside the grid")

  P = np.zeros((grid_size, grid_size, 4), dtype=float)
  if resolved == "uniform":
    P[x0, y0, :] = 0.25
    stored_orientation = "uniform"
  else:
    P[x0, y0, ORIENTATION_INDEX[resolved]] = 1.0
    stored_orientation = resolved

  grid = OrientationGrid.from_array(P)
  grid.initial_orientation = stored_orientation
  return grid


def gaussian_initial_state(
  grid_size: int,
  *,
  center: Optional[Tuple[int, int]] = None,
  width: float = 10.0,
  orientation: str = "right",
  rng: Optional[np.random.Generator] = None,
) -> OrientationGrid:
  """Gaussian spatial envelope with fixed or uniform orientation weight."""
  if center is None:
    center = (grid_size // 2, grid_size // 2)

  resolved = _resolve_orientation(orientation, rng=rng)
  x0, y0 = center
  xs = np.arange(grid_size)
  ys = np.arange(grid_size)
  X, Y = np.meshgrid(xs, ys, indexing="ij")
  weights = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / width)

  P = np.zeros((grid_size, grid_size, 4), dtype=float)
  if resolved == "uniform":
    for idx in range(4):
      P[:, :, idx] = 0.25 * weights
    stored_orientation = "uniform"
  else:
    P[:, :, ORIENTATION_INDEX[resolved]] = weights
    stored_orientation = resolved

  grid = OrientationGrid.from_array(P)
  grid.initial_orientation = stored_orientation
  grid.normalize()
  return grid


def apply_coin(grid: OrientationGrid, coin: np.ndarray) -> None:
  """Apply coin operator C at every lattice site: P' = C @ P.

  ``coin`` may be a global (4, 4) matrix or a site-local (N, N, 4, 4) field.
  """
  P = grid.to_array()
  coin = normalize_coin_input(coin, grid.grid_size)
  _write_orientation_array(grid, _apply_coin_to_P(P, coin))


def _boundary_mask(grid_size: int) -> np.ndarray:
  x = np.arange(grid_size)[:, None]
  y = np.arange(grid_size)[None, :]
  return (x == 0) | (x == grid_size - 1) | (y == 0) | (y == grid_size - 1)


def apply_coin_reflective_infinite_force(grid: OrientationGrid, coin: np.ndarray) -> None:
  """Julia-style infinite-force walls: reflection coin on the boundary, ACP coin inside.

  The reflection matrix swaps right↔left and up↔down, matching src/boundary_coins.jl.
  Interior sites use a global (4, 4) coin or a site-local (N, N, 4, 4) field.
  """
  P = grid.to_array()
  coin = normalize_coin_input(coin, grid.grid_size)
  P_coin = _apply_coin_to_P(P, coin)
  P_refl = np.einsum("ij,xyj->xyi", REFLECTION_MATRIX_4, P)
  mask = _boundary_mask(grid.grid_size)[:, :, None]
  _write_orientation_array(grid, np.where(mask, P_refl, P_coin))


def _write_orientation_array(grid: OrientationGrid, P: np.ndarray) -> None:
  grid.df[list(ORIENTATION_COLS)] = P.reshape(-1, 4)
  _update_rho(grid.df)


def _shift_slice_periodic(slice_2d: np.ndarray, dx: int, dy: int) -> np.ndarray:
  out = slice_2d
  if dx:
    out = np.roll(out, shift=dx, axis=0)
  if dy:
    out = np.roll(out, shift=dy, axis=1)
  return out


def _shift_mesh(grid_size: int) -> Tuple[np.ndarray, np.ndarray]:
  xs = np.arange(grid_size)
  ys = np.arange(grid_size)
  return np.meshgrid(xs, ys, indexing="ij")


def _shift_slice_open(slice_2d: np.ndarray, dx: int, dy: int) -> np.ndarray:
  n = slice_2d.shape[0]
  x_grid, y_grid = _shift_mesh(n)
  sx = x_grid - dx
  sy = y_grid - dy
  mask = (sx >= 0) & (sx < n) & (sy >= 0) & (sy < n)
  out = np.zeros_like(slice_2d)
  out[mask] = slice_2d[sx[mask], sy[mask]]
  return out


def _shift_slice_reflecting(P: np.ndarray, coin: np.ndarray, reflection_coin_type: ReflectionCoinType = "reflecting") -> np.ndarray:
  """No-flux stay-put walls, with an extra coin on blocked hops.

  If r + U_d is inside, P_d is transported there with heading d. If the hop
  would leave the box, that mass stays at r, but a stay-put on the lattice
  is treated as two operations in one dt: apply the wall coin to the
  blocked component, then add the coined 4-vector at r. Incoming interior
  mass is still added. Total probability is conserved.

  Wall-coin options via ``reflection_coin_type``:
    - ``reflecting`` / ``anti_reflecting``: bulk coin (or its transpose)
    - ``reflective_infinite_force``: hard flip right↔left, up↔down
    - ``randomized``: uniform random reorientation
    - ``stay``: keep the blocked orientation unchanged (identity)
    - ``scatter``: split blocked heading equally into the other three dirs

  With a site-local coin field, blocked mass at r uses that site's wall coin
  (except global types ``randomized``, ``reflective_infinite_force``,
  ``stay``, ``scatter``).
  """
  wall_coin = _wall_coin_field(coin, reflection_coin_type)
  n = P.shape[0]
  x_grid, y_grid = _shift_mesh(n)
  out = np.zeros_like(P)
  blocked = np.zeros_like(P)
  for d, (dx, dy) in enumerate(SHIFTS):
    dest_x = x_grid + dx
    dest_y = y_grid + dy
    inside = (dest_x >= 0) & (dest_x < n) & (dest_y >= 0) & (dest_y < n)
    channel = P[:, :, d]
    out[dest_x[inside], dest_y[inside], d] += channel[inside]
    blocked[:, :, d][~inside] = channel[~inside]
  out += _apply_coin_to_P(blocked, wall_coin)
  # out += _apply_coin_to_P(_apply_coin_to_P(blocked, wall_coin), wall_coin)  # apply coin twice
  return out


def apply_shift(
  grid: OrientationGrid,
  boundary: BoundaryMode,
  coin: Optional[np.ndarray] = None,
  reflection_coin_type: ReflectionCoinType = "reflecting",
) -> None:
  """Transport each orientation channel after the coin step."""
  P = grid.to_array()

  if boundary in ("periodic", "reflective_infinite_force"):
    shifted = np.zeros_like(P)
    for d, (dx, dy) in enumerate(SHIFTS):
      shifted[:, :, d] = _shift_slice_periodic(P[:, :, d], dx, dy)
  elif boundary == "open":
    shifted = np.zeros_like(P)
    for d, (dx, dy) in enumerate(SHIFTS):
      shifted[:, :, d] = _shift_slice_open(P[:, :, d], dx, dy)
  elif boundary == "reflecting":
    if coin is None:
      raise ValueError("reflecting shift requires the bulk coin")
    if reflection_coin_type is None:
      raise ValueError("reflecting shift requires the reflection coin type")
    shifted = _shift_slice_reflecting(P, coin, reflection_coin_type)
  else:
    raise ValueError(f"Unknown boundary mode: {boundary}")

  _write_orientation_array(grid, shifted)


def _shift_absorbing_reflecting_defects(
  P: np.ndarray,
  coin: np.ndarray,
  obstacle_mask: np.ndarray,
  reflection_coin_type: ReflectionCoinType = "reflecting",
) -> np.ndarray:
  """Absorbing outer wall and reflecting interior defects.

  A hop into a free in-box cell is transported. A hop into a defect stays at
  the origin and is passed through the wall coin. ``reflecting`` reapplies
  the bulk coin. ``specular`` reverses that heading, matching the continuum
  face reflection. A hop that leaves the box is removed.
  """
  if obstacle_mask.shape != P.shape[:2]:
    raise ValueError(
      f"obstacle_mask shape {obstacle_mask.shape} does not match grid {P.shape[:2]}"
    )
  if obstacle_mask.dtype != bool:
    obstacle_mask = obstacle_mask.astype(bool)

  wall_coin = _wall_coin_field(coin, reflection_coin_type)
  n = P.shape[0]
  x_grid, y_grid = _shift_mesh(n)
  out = np.zeros_like(P)
  blocked = np.zeros_like(P)
  for d, (dx, dy) in enumerate(SHIFTS):
    dest_x = x_grid + dx
    dest_y = y_grid + dy
    in_box = (dest_x >= 0) & (dest_x < n) & (dest_y >= 0) & (dest_y < n)
    dest_x_c = np.clip(dest_x, 0, n - 1)
    dest_y_c = np.clip(dest_y, 0, n - 1)
    hits_defect = in_box & obstacle_mask[dest_x_c, dest_y_c]
    free = in_box & ~hits_defect
    channel = P[:, :, d]
    if np.any(free):
      np.add.at(out[:, :, d], (dest_x[free], dest_y[free]), channel[free])
    if np.any(hits_defect):
      blocked[:, :, d][hits_defect] = channel[hits_defect]
  out += _apply_coin_to_P(blocked, wall_coin)
  out[obstacle_mask] = 0.0
  return out


def step_absorbing_reflecting_defects(
  P: np.ndarray,
  coin: np.ndarray,
  obstacle_mask: np.ndarray,
  reflection_coin_type: ReflectionCoinType = "reflecting",
) -> np.ndarray:
  """One lattice step on an orientation field with defects.

  Coin, then shift. Defect cells are cleared before and after the step.
  ``open`` / ``reflecting`` evolution is not used here.
  """
  if obstacle_mask.shape != P.shape[:2]:
    raise ValueError(
      f"obstacle_mask shape {obstacle_mask.shape} does not match grid {P.shape[:2]}"
    )
  obstacle_mask = np.asarray(obstacle_mask, dtype=bool)
  coin = normalize_coin_input(coin, P.shape[0])
  field = P.copy()
  field[obstacle_mask] = 0.0
  field = _apply_coin_to_P(field, coin)
  field[obstacle_mask] = 0.0
  return _shift_absorbing_reflecting_defects(
    field, coin, obstacle_mask, reflection_coin_type
  )


def evolve_obstacles(
  grid: OrientationGrid,
  coin: np.ndarray,
  obstacle_mask: np.ndarray,
  max_time: int,
  save_times: Sequence[int],
  reflection_coin_type: ReflectionCoinType = "reflecting",
) -> Tuple[OrientationGrid, dict[int, pd.DataFrame], np.ndarray, np.ndarray]:
  """Propagate with absorbing outer walls and reflecting defects.

  Returns the final grid, snapshots at ``save_times``, survival ``S(t)`` of
  length ``max_time + 1``, and the spatial density at every integer time with
  shape ``(max_time + 1, N, N)``.
  """
  obstacle_mask = np.asarray(obstacle_mask, dtype=bool)
  if obstacle_mask.shape != (grid.grid_size, grid.grid_size):
    raise ValueError(
      f"obstacle_mask must have shape {(grid.grid_size, grid.grid_size)}"
    )
  coin = normalize_coin_input(coin, grid.grid_size)
  save_set = {int(t) for t in save_times}
  P = np.array(grid.to_array(), dtype=float, copy=True)
  P[obstacle_mask] = 0.0

  snapshots: dict[int, pd.DataFrame] = {}
  survival = np.empty(max_time + 1, dtype=float)
  rho_series = np.empty((max_time + 1, grid.grid_size, grid.grid_size), dtype=float)
  survival[0] = float(P.sum())
  rho_series[0] = P.sum(axis=-1)
  if 0 in save_set:
    snapshots[0] = OrientationGrid.from_array(P, time=0).snapshot_dataframe()

  for step in range(1, max_time + 1):
    P = step_absorbing_reflecting_defects(
      P, coin, obstacle_mask, reflection_coin_type
    )
    survival[step] = float(P.sum())
    rho_series[step] = P.sum(axis=-1)
    if step in save_set:
      snapshots[step] = OrientationGrid.from_array(P, time=step).snapshot_dataframe()

  final = OrientationGrid.from_array(P, time=max_time)
  return final, snapshots, survival, rho_series


def evolve_step(grid: OrientationGrid, coin: np.ndarray, boundary: BoundaryMode, reflection_coin_type: ReflectionCoinType = "reflecting") -> None:
  """One discrete time step: coin then shift.

  ``coin`` may be a global (4, 4) matrix or a site-local (N, N, 4, 4) field.
  ``reflective_infinite_force`` uses a reflection coin on walls, then periodic
  transport (Julia src). ``reflecting`` uses the bulk coin everywhere, then
  stay-put no-flux walls; blocked (outward) mass is coined once more before
  being added back at the wall.
  """
  coin = normalize_coin_input(coin, grid.grid_size)
  if boundary == "reflective_infinite_force":
    apply_coin_reflective_infinite_force(grid, coin)
  else:
    apply_coin(grid, coin)
  apply_shift(grid, boundary, coin=coin, reflection_coin_type=reflection_coin_type)
  grid.time += 1


def evolve(
  grid: OrientationGrid,
  coin: np.ndarray,
  max_time: int,
  save_times: Sequence[int],
  boundary: BoundaryMode,
  reflection_coin_type: ReflectionCoinType = "reflecting",
) -> Tuple[OrientationGrid, dict[int, pd.DataFrame]]:
  """Propagate the distribution for max_time steps; save snapshots at save_times.

  ``coin`` may be a global (4, 4) matrix or a site-local (N, N, 4, 4) field.
  """
  coin = normalize_coin_input(coin, grid.grid_size)
  save_set = set(int(t) for t in save_times)
  snapshots: dict[int, pd.DataFrame] = {}

  if 0 in save_set:
    snapshots[0] = grid.snapshot_dataframe()

  working = grid.copy()
  for _ in range(max_time):
    evolve_step(working, coin, boundary, reflection_coin_type)
    if working.time in save_set:
      snapshots[working.time] = working.snapshot_dataframe()

  return working, snapshots


def snapshot_to_rho_array(snapshot: pd.DataFrame, grid_size: int) -> np.ndarray:
  """Convert a snapshot DataFrame to a spatial rho array with shape (N, N)."""
  return snapshot["rho"].unstack(level="y").to_numpy(dtype=float)


def snapshot_to_orientation_array(snapshot: pd.DataFrame, grid_size: int) -> np.ndarray:
  """Return orientation field P with shape (N, N, 4) in ORIENTATION_COLS order."""
  stacked = np.stack(
    [
      snapshot[col].unstack(level="y").to_numpy(dtype=float)
      for col in ORIENTATION_COLS
    ],
    axis=-1,
  )
  return stacked


def flux_from_orientation(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Probability current from the 4-orientation field.

  With state order (right, up, left, down):
    J_x = P_right - P_left
    J_y = P_up - P_down
  Returns (J_x, J_y, |J|), each with shape (N, N).
  """
  jx = P[:, :, 0] - P[:, :, 2]
  jy = P[:, :, 1] - P[:, :, 3]
  j_mag = np.hypot(jx, jy)
  return jx, jy, j_mag


def msd_from_rho(
  rho: np.ndarray,
  origin: Tuple[int, int],
  *,
  grid_size: Optional[int] = None,
) -> float:
  """MSD = sum_{x,y} rho(x,y) * [(x-x0)^2 + (y-y0)^2]."""
  n = grid_size if grid_size is not None else rho.shape[0]
  x0, y0 = origin
  xs = np.arange(n)
  ys = np.arange(n)
  X, Y = np.meshgrid(xs, ys, indexing="ij")
  r_sq = (X - x0) ** 2 + (Y - y0) ** 2
  return float(np.sum(rho * r_sq))


def msd_from_snapshots(
  snapshots: dict[int, pd.DataFrame],
  origin: Tuple[int, int],
  *,
  grid_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
  """Compute MSD at each saved time from snapshot DataFrames."""
  times = np.array(sorted(snapshots), dtype=int)
  msd_values = np.array([
    msd_from_rho(
      snapshot_to_rho_array(snapshots[t], grid_size),
      origin,
      grid_size=grid_size,
    )
    for t in times
  ])
  return times, msd_values
