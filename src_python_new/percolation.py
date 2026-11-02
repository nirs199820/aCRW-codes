"""Percolation grids and stochastic walk simulations."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .coins import (
  CHIRAL_MATRIX,
  DIRECTION_TABLE,
  PERSISTENCE_MATRIX,
  REFLECTION_MATRIX,
  sample_coin_direction,
  sample_coin_direction_batch,
)
from .directions import random_direction


def mod1(n: int, m: int) -> int:
  """Julia mod1: wrap n into range 1..m."""
  return (n - 1) % m + 1


@dataclass
class PercolationGrid:
  pattern: np.ndarray
  cluster: np.ndarray
  initialconditions: List[Tuple[int, int]]
  x0: int
  y0: int
  xt: int
  yt: int
  scratch: np.ndarray


def create_percolation_pattern(*, grid_size: int, seed: int, q: float) -> np.ndarray:
  rng = np.random.default_rng(seed)
  pattern = rng.random((grid_size, grid_size)) < q
  return pattern.astype(int)


def create_boundary_pattern(*, grid_size: int) -> np.ndarray:
  pattern = np.ones((grid_size, grid_size), dtype=int)
  pattern[0, :] = 0
  pattern[-1, :] = 0
  pattern[:, 0] = 0
  pattern[:, -1] = 0
  return pattern


def find_largest_cluster(pattern: np.ndarray) -> np.ndarray:
  rows, cols = pattern.shape
  labels = np.zeros((rows, cols), dtype=int)
  parent: List[int] = [0]

  def find_root(x: int) -> int:
    while parent[x] != x:
      parent[x] = parent[parent[x]]
      x = parent[x]
    return x

  def union(a: int, b: int) -> int:
    ra, rb = find_root(a), find_root(b)
    if ra < rb:
      parent[rb] = ra
    elif rb < ra:
      parent[ra] = rb
    return min(ra, rb)

  next_label = 1
  for i in range(rows):
    for j in range(cols):
      if pattern[i, j] == 0:
        continue
      left = labels[i, j - 1] if j > 0 else 0
      above = labels[i - 1, j] if i > 0 else 0
      if left == 0 and above == 0:
        parent.append(next_label)
        labels[i, j] = next_label
        next_label += 1
      elif left == 0:
        labels[i, j] = find_root(above)
      elif above == 0:
        labels[i, j] = find_root(left)
      else:
        labels[i, j] = union(left, above)

  for j in range(cols):
    if pattern[0, j] == 1 and pattern[rows - 1, j] == 1:
      union(labels[0, j], labels[rows - 1, j])

  for i in range(rows):
    if pattern[i, 0] == 1 and pattern[i, cols - 1] == 1:
      union(labels[i, 0], labels[i, cols - 1])

  counts: dict[int, int] = {}
  for i in range(rows):
    for j in range(cols):
      if labels[i, j] == 0:
        continue
      root = find_root(labels[i, j])
      labels[i, j] = root
      counts[root] = counts.get(root, 0) + 1

  if not counts:
    return np.zeros((rows, cols), dtype=int)

  largest_label = max(counts, key=counts.get)
  return (labels == largest_label).astype(int)


def create_percolation_grid(*, grid_size: int, seed: int, q: float) -> PercolationGrid:
  if q != 0:
    pattern = create_percolation_pattern(grid_size=grid_size, seed=seed, q=q)
    cluster = find_largest_cluster(pattern)
    initialconditions = [(i + 1, j + 1) for i, j in zip(*np.where(cluster == 1))]
    x0, y0 = random.choice(initialconditions)
    scratch = np.full((grid_size, grid_size), -1, dtype=int)
  else:
    pattern = create_boundary_pattern(grid_size=grid_size)
    cluster = find_largest_cluster(pattern)
    initialconditions = random.sample(range(1, grid_size - 1), 2)
    x0, y0 = initialconditions
    scratch = np.full((grid_size, grid_size), -1, dtype=int)
  return PercolationGrid(pattern, cluster, initialconditions, x0, y0, 0, 0, scratch)


def get_new_initial_condition(g: PercolationGrid) -> PercolationGrid:
  x0, y0 = random.choice(g.initialconditions)
  rows, cols = g.cluster.shape
  scratch = np.full((rows, cols), -1, dtype=int)
  return PercolationGrid(g.pattern, g.cluster, g.initialconditions, x0, y0, g.xt, g.yt, scratch)


def find_target(g: PercolationGrid, n_hops: int) -> bool:
  rows, cols = g.cluster.shape
  dist = g.scratch
  dist.fill(-1)
  dist[g.x0 - 1, g.y0 - 1] = 0
  queue: deque[Tuple[int, int]] = deque([(g.x0, g.y0)])

  while queue:
    i, j = queue.popleft()
    if dist[i - 1, j - 1] == n_hops:
      continue
    for ni, nj in [
      (mod1(i - 1, rows), j),
      (mod1(i + 1, rows), j),
      (i, mod1(j - 1, cols)),
      (i, mod1(j + 1, cols)),
    ]:
      if g.cluster[ni - 1, nj - 1] == 1 and dist[ni - 1, nj - 1] == -1:
        dist[ni - 1, nj - 1] = dist[i - 1, j - 1] + 1
        queue.append((ni, nj))

  targets = [(i + 1, j + 1) for i, j in zip(*np.where(dist == n_hops))]
  if not targets:
    return False
  g.xt, g.yt = random.choice(targets)
  return True


def set_new_initial_and_target(g: PercolationGrid, x0: int, y0: int, n_hops: int) -> bool:
  g.x0 = x0
  g.y0 = y0
  return find_target(g, n_hops)


def make_coin(p: float) -> Tuple[np.ndarray, np.ndarray]:
  return make_general_coin(p_q=1.0 - p, p_xi=p)


def make_general_coin(p_q: float, p_xi: float) -> Tuple[np.ndarray, np.ndarray]:
  if abs(p_q) + abs(p_xi) > 1.0:
    raise ValueError("|p_q| + |p_xi| must be <= 1")
  standard_coin = p_q * PERSISTENCE_MATRIX + p_xi * CHIRAL_MATRIX
  return standard_coin, REFLECTION_MATRIX.copy()


def log_save_times(T: int) -> List[int]:
  pts = {0}
  decade = 1
  while decade <= T:
    for k in range(1, 10):
      t = k * decade
      if t <= T:
        pts.add(t)
    decade *= 10
  return sorted(pts)


def logspace_save_times(T: int, n_per_decade: int = 10) -> List[int]:
  log_T = np.log10(T)
  n = round(log_T * n_per_decade)
  times = np.unique(np.round(10 ** np.linspace(0, log_T, n + 1)).astype(int))
  times = [int(t) for t in times if t >= 1]
  return [0] + times


def simulate_percolation_walk(
  g: PercolationGrid,
  standard_coin: np.ndarray,
  reflection_matrix: np.ndarray,
  save_times: List[int],
  *,
  x0: int,
  y0: int,
  max_steps: int,
  return_directions: bool = False,
) -> Tuple[List[int], List[int]]:
  rows, cols = g.pattern.shape

  x, y = x0, y0
  ux, uy = x0, y0
  direction = random_direction()

  n_save = len(save_times)
  traj_ux = [0] * n_save
  traj_uy = [0] * n_save
  traj_ux[0] = ux
  traj_uy[0] = uy
  traj_directions_x = [0] * n_save
  traj_directions_y = [0] * n_save
  save_ptr = 1

  for step in range(1, max_steps + 1):
    coin = standard_coin if g.pattern[x - 1, y - 1] == 1 else reflection_matrix
    direction = sample_coin_direction(coin, direction)

    dx, dy = direction
    x = mod1(x + dx, rows)
    y = mod1(y + dy, cols)
    ux += dx
    uy += dy

    if save_ptr < n_save and step == save_times[save_ptr]:
      traj_ux[save_ptr] = ux
      traj_uy[save_ptr] = uy
      traj_directions_x[save_ptr] = dx
      traj_directions_y[save_ptr] = dy
      save_ptr += 1

  if return_directions:
    return traj_ux, traj_uy, traj_directions_x, traj_directions_y
  return traj_ux, traj_uy


def simulate_percolation_walk_time_dependent_coins(
  g: PercolationGrid,
  time_dependent_coins: List[np.ndarray],
  reflection_matrix: np.ndarray,
  save_times: List[int],
  *,
  x0: int,
  y0: int,
  max_steps: int,
  return_directions: bool = False,
) -> Tuple[List[int], List[int]]:
  rows, cols = g.pattern.shape

  x, y = x0, y0
  ux, uy = x0, y0
  direction = random_direction()

  n_save = len(save_times)
  traj_ux = [0] * n_save
  traj_uy = [0] * n_save
  traj_ux[0] = ux
  traj_uy[0] = uy
  traj_directions_x = [0] * n_save
  traj_directions_y = [0] * n_save
  save_ptr = 1

  for step in range(1, max_steps + 1):
    coin = time_dependent_coins[step - 1] if g.pattern[x - 1, y - 1] == 1 else reflection_matrix
    direction = sample_coin_direction(coin, direction)

    dx, dy = direction
    x = mod1(x + dx, rows)
    y = mod1(y + dy, cols)
    ux += dx
    uy += dy

    if save_ptr < n_save and step == save_times[save_ptr]:
      traj_ux[save_ptr] = ux
      traj_uy[save_ptr] = uy
      traj_directions_x[save_ptr] = dx
      traj_directions_y[save_ptr] = dy
      save_ptr += 1

  if return_directions:
    return traj_ux, traj_uy, traj_directions_x, traj_directions_y
  return traj_ux, traj_uy


def simulate_percolation_ensemble_time_dependent_coins(
  g: PercolationGrid,
  time_dependent_coins: List[np.ndarray],
  reflection_matrix: np.ndarray,
  save_times: List[int],
  *,
  N_particles: int,
  x0: int,
  y0: int,
  max_steps: int,
  return_directions: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
  n_save = len(save_times)
  traj_ux = np.zeros((N_particles, n_save), dtype=int)
  traj_uy = np.zeros((N_particles, n_save), dtype=int)

  if return_directions:
    traj_directions_x = np.zeros((N_particles, n_save), dtype=int)
    traj_directions_y = np.zeros((N_particles, n_save), dtype=int)
    for i in range(N_particles):
      ux, uy, dx, dy = simulate_percolation_walk_time_dependent_coins(
        g,
        time_dependent_coins,
        reflection_matrix,
        save_times,
        x0=x0,
        y0=y0,
        max_steps=max_steps,
        return_directions=True,
      )
      traj_ux[i] = ux
      traj_uy[i] = uy
      traj_directions_x[i] = dx
      traj_directions_y[i] = dy
    return traj_ux, traj_uy, traj_directions_x, traj_directions_y

  for i in range(N_particles):
    ux, uy = simulate_percolation_walk_time_dependent_coins(
      g,
      time_dependent_coins,
      reflection_matrix,
      save_times,
      x0=x0,
      y0=y0,
      max_steps=max_steps,
      return_directions=False,
    )
    traj_ux[i] = ux
    traj_uy[i] = uy
  return traj_ux, traj_uy


def simulate_percolation_ensemble_time_dependent_coins_batched(
  g: PercolationGrid,
  time_dependent_coins: List[np.ndarray],
  reflection_matrix: np.ndarray,
  save_times: List[int],
  *,
  N_particles: int,
  x0: int,
  y0: int,
  max_steps: int,
  return_directions: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
  rows, cols = g.pattern.shape
  n_save = len(save_times)
  rng = np.random.default_rng()

  x = np.full(N_particles, x0, dtype=int)
  y = np.full(N_particles, y0, dtype=int)
  ux = np.full(N_particles, x0, dtype=int)
  uy = np.full(N_particles, y0, dtype=int)
  dir_idx = rng.integers(0, 4, size=N_particles)

  traj_ux = np.zeros((N_particles, n_save), dtype=int)
  traj_uy = np.zeros((N_particles, n_save), dtype=int)
  traj_ux[:, 0] = x0
  traj_uy[:, 0] = y0
  traj_directions_x = None
  traj_directions_y = None
  if return_directions:
    traj_directions_x = np.zeros((N_particles, n_save), dtype=int)
    traj_directions_y = np.zeros((N_particles, n_save), dtype=int)

  save_ptr = 1
  for step in range(1, max_steps + 1):
    coin_open = time_dependent_coins[step - 1]
    on_open = g.pattern[x - 1, y - 1] == 1
    dir_idx = sample_coin_direction_batch(
      coin_open,
      reflection_matrix,
      on_open,
      dir_idx,
      rng,
    )

    dx = DIRECTION_TABLE[dir_idx, 0]
    dy = DIRECTION_TABLE[dir_idx, 1]
    x = (x - 1 + dx) % rows + 1
    y = (y - 1 + dy) % cols + 1
    ux = ux + dx
    uy = uy + dy

    if save_ptr < n_save and step == save_times[save_ptr]:
      traj_ux[:, save_ptr] = ux
      traj_uy[:, save_ptr] = uy
      if return_directions:
        traj_directions_x[:, save_ptr] = dx
        traj_directions_y[:, save_ptr] = dy
      save_ptr += 1

  if return_directions:
    return traj_ux, traj_uy, traj_directions_x, traj_directions_y
  return traj_ux, traj_uy


def simulate_distinct_sites(
  g: PercolationGrid,
  standard_coin: np.ndarray,
  reflection_matrix: np.ndarray,
  save_times: List[int],
  *,
  x0: int,
  y0: int,
  max_steps: int,
) -> List[int]:
  rows, cols = g.pattern.shape

  x, y = x0, y0
  direction = random_direction()

  visited = np.zeros((rows, cols), dtype=bool)
  visited[x - 1, y - 1] = True
  n_visited = 1

  n_save = len(save_times)
  traj_S = [0] * n_save
  traj_S[0] = n_visited
  save_ptr = 1

  for step in range(1, max_steps + 1):
    coin = standard_coin if g.pattern[x - 1, y - 1] == 1 else reflection_matrix
    direction = sample_coin_direction(coin, direction)

    dx, dy = direction
    x = mod1(x + dx, rows)
    y = mod1(y + dy, cols)

    if not visited[x - 1, y - 1]:
      visited[x - 1, y - 1] = True
      n_visited += 1

    if save_ptr < n_save and step == save_times[save_ptr]:
      traj_S[save_ptr] = n_visited
      save_ptr += 1

  return traj_S


def compute_distances(g: PercolationGrid, max_dist: int = 2**31 - 1) -> int:
  rows, cols = g.cluster.shape
  dist = g.scratch
  dist.fill(-1)
  dist[g.x0 - 1, g.y0 - 1] = 0
  queue: deque[Tuple[int, int]] = deque([(g.x0, g.y0)])
  max_reached = 0

  while queue:
    i, j = queue.popleft()
    d = dist[i - 1, j - 1]
    if d >= max_dist:
      continue
    for ni, nj in [
      (mod1(i - 1, rows), j),
      (mod1(i + 1, rows), j),
      (i, mod1(j - 1, cols)),
      (i, mod1(j + 1, cols)),
    ]:
      if g.cluster[ni - 1, nj - 1] == 1 and dist[ni - 1, nj - 1] == -1:
        dist[ni - 1, nj - 1] = d + 1
        if d + 1 > max_reached:
          max_reached = d + 1
        queue.append((ni, nj))

  return max_reached


def simulate_first_passage_walk(
  g: PercolationGrid,
  standard_coin: np.ndarray,
  reflection_matrix: np.ndarray,
  x0: int,
  y0: int,
  r_values: List[int],
  max_steps: int,
) -> List[int]:
  rows, cols = g.pattern.shape
  n_r = len(r_values)

  g.x0 = x0
  g.y0 = y0
  max_reached = compute_distances(g)
  dist = g.scratch

  fpt = [0] * n_r
  done = [False] * n_r
  n_done = 0

  for k, rv in enumerate(r_values):
    if rv > max_reached:
      done[k] = True
      n_done += 1

  if n_done == n_r:
    return fpt

  x, y = x0, y0
  direction = random_direction()

  for step in range(1, max_steps + 1):
    coin = standard_coin if g.pattern[x - 1, y - 1] == 1 else reflection_matrix
    direction = sample_coin_direction(coin, direction)

    dx, dy = direction
    x = mod1(x + dx, rows)
    y = mod1(y + dy, cols)

    d = dist[x - 1, y - 1]
    if d <= 0:
      continue

    for k, rv in enumerate(r_values):
      if not done[k] and d >= rv:
        fpt[k] = step
        done[k] = True
        n_done += 1

    if n_done == n_r:
      break

  for k in range(n_r):
    if not done[k]:
      fpt[k] = -1

  return fpt


def simulate_hitting_time_walk(
  g: PercolationGrid,
  standard_coin: np.ndarray,
  reflection_matrix: np.ndarray,
  x0: int,
  y0: int,
  n_hops: int,
  max_steps: int,
) -> int:
  rows, cols = g.pattern.shape
  if not set_new_initial_and_target(g, x0, y0, n_hops):
    return 0

  x, y = g.x0, g.y0
  direction = random_direction()
  hitting_time = -1

  for step in range(1, max_steps + 1):
    coin = standard_coin if g.pattern[x - 1, y - 1] == 1 else reflection_matrix
    direction = sample_coin_direction(coin, direction)

    dx, dy = direction
    x = mod1(x + dx, rows)
    y = mod1(y + dy, cols)

    if x == g.xt and y == g.yt:
      hitting_time = step
      break

  return hitting_time
