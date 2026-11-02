"""Conservative finite-volume solver for a two-dimensional chiral ABP.

    p_t = D Laplacian(p) - v0 u(phi).grad(p) + Dr p_phiphi - omega p_phi

The cell-centered spatial domain is [0, N*ell]^2; phi is periodic. Stored
values are probability MASSES per spatial cell and angular bin, not density
samples. Outer spatial faces absorb probability. At defects, the active
transport reflects the heading specularly (vertical: pi-phi; horizontal:
-phi), while translational diffusion has zero normal flux. This is the
specified hybrid collision rule, not the usual torque-free steric ABP wall.

Space and angle use conservative minmod reconstruction for advection and
centered diffusive face fluxes. All directions are advanced together with
SSP RK2: second order in time, second order in space/angle in smooth regions,
and locally first order where reconstruction is limited or at walls. There
is no Fourier angular interpolation, clipping, or probability renormalization.
Automatic substeps enforce a sufficient positivity bound, including absorbing
boundary diffusion. The angular mesh must have an even number of bins >= 4.

Public function names and the original output keys are retained. Additional
outputs include internal_dt, substeps_per_dt, minimum_probability, and
face_flux_snapshots. The latter are TOTAL numerical face currents in units
of probability/time; flux_snapshots retains the original ACTIVE cell-integrated
moment, in units of probability*length/time.

The initial source is uniform in the selected lattice cell. By default its
heading is isotropic; initial_orientation can select a directed heading or
custom angular weights. The source is not a spatial delta function and does
not shrink as refine increases.
NumPy is required. Optional Numba compiles the SAME spatial/angular fluxes
and SSP RK2 steps into fused parallel loops over free cells. Install a
mutually compatible NumPy/Numba pair in your Python environment to use it.
The first accelerated call compiles kernels; later calls reuse the cache.
No coarser grid, larger time steps, fast-math, or change of physics is used.
backend="auto" selects Numba when importable, otherwise warns and uses NumPy.
backend="numba" requires acceleration; backend="numpy" uses the reference path.
"""

from __future__ import annotations

import math
import operator
import warnings
from typing import Optional, Sequence

import numpy as np


# Loaded lazily so this remains usable with NumPy alone.
prange = range
_NUMBA_KERNELS = None


def _slopes_impl(p, neighbors, sx, sy, sa, spatial_active, angular_active):
    """Compute the same minmod slopes as the reference implementation."""
    na = p.shape[1]
    for node in prange(p.shape[0]):
        left, right, down, up = neighbors[node]
        for a in range(na):
            q = p[node, a]
            x = 0.0
            if spatial_active and left >= 0 and right >= 0:
                dl, dr = q - p[left, a], p[right, a] - q
                if dl > 0 and dr > 0:
                    x = min(dl, dr)
                elif dl < 0 and dr < 0:
                    x = -min(-dl, -dr)
            sx[node, a] = x
            y = 0.0
            if spatial_active and down >= 0 and up >= 0:
                dl, dr = q - p[down, a], p[up, a] - q
                if dl > 0 and dr > 0:
                    y = min(dl, dr)
                elif dl < 0 and dr < 0:
                    y = -min(-dl, -dr)
            sy[node, a] = y
            angle = 0.0
            if angular_active:
                am = a - 1 if a else na - 1
                ap = a + 1 if a + 1 < na else 0
                dl, dr = q - p[node, am], p[node, ap] - q
                if dl > 0 and dr > 0:
                    angle = min(dl, dr)
                elif dl < 0 and dr < 0:
                    angle = -min(-dl, -dr)
            sa[node, a] = angle


def _update_impl(p, base, neighbors, sx, sy, sa, cx, cy, rx, ry,
                 spatial_diffusion, angular_diffusion, angular_speed,
                 h, finish, out, minima):
    """Gather flux divergences; each thread writes its own cell only."""
    na = p.shape[1]
    for node in prange(p.shape[0]):
        row_min = math.inf
        for a in range(na):
            q = p[node, a]
            rate = 0.0
            for axis in range(2):
                if axis == 0:
                    c, reflected = cx[a], rx[a]
                    cr = cx[reflected]
                    slopes = sx
                else:
                    c, reflected = cy[a], ry[a]
                    cr = cy[reflected]
                    slopes = sy
                left, right = neighbors[node, 2*axis], neighbors[node, 2*axis+1]
                if left >= 0:
                    state = (p[left, a] + 0.5*slopes[left, a]) if c >= 0 else (q - 0.5*slopes[node, a])
                    rate += c*state + spatial_diffusion*(p[left, a] - q)
                elif left == -2:  # absorbing exterior
                    rate -= (max(-c, 0.0) + 2*spatial_diffusion)*q
                else:  # defect: specular angular-bin transfer
                    rate += max(-cr, 0.0)*p[node, reflected] - max(-c, 0.0)*q
                if right >= 0:
                    state = (q + 0.5*slopes[node, a]) if c >= 0 else (p[right, a] - 0.5*slopes[right, a])
                    rate -= c*state + spatial_diffusion*(q - p[right, a])
                elif right == -2:
                    rate -= (max(c, 0.0) + 2*spatial_diffusion)*q
                else:
                    rate += max(cr, 0.0)*p[node, reflected] - max(c, 0.0)*q

            if angular_speed != 0.0 or angular_diffusion != 0.0:
                am = a - 1 if a else na - 1
                ap = a + 1 if a + 1 < na else 0
                if angular_speed >= 0:
                    right_flux = angular_speed*(q + 0.5*sa[node, a])
                    left_flux = angular_speed*(p[node, am] + 0.5*sa[node, am])
                else:
                    right_flux = angular_speed*(p[node, ap] - 0.5*sa[node, ap])
                    left_flux = angular_speed*(q - 0.5*sa[node, a])
                right_flux += angular_diffusion*(q - p[node, ap])
                left_flux += angular_diffusion*(p[node, am] - q)
                rate += left_flux - right_flux
            advanced = q + h*rate
            if advanced < row_min:
                row_min = advanced
            out[node, a] = 0.5*base[node, a] + 0.5*advanced if finish else advanced
        minima[node] = row_min


def _load_numba():
    global prange, _NUMBA_KERNELS
    if _NUMBA_KERNELS is None:
        import numba
        prange = numba.prange
        _NUMBA_KERNELS = (
            numba,
            numba.njit(parallel=True, cache=True, nogil=True)(_slopes_impl),
            numba.njit(parallel=True, cache=True, nogil=True)(_update_impl),
        )
    return _NUMBA_KERNELS


class _CompiledStepper:
    """Keep only free cells and reuse every work buffer across substeps."""
    def __init__(self, prob, free, vx, vy, rx, ry, dx, dphi,
                 diffusion, d_r, omega, kernels, workers):
        self.numba, self.slopes, self.update = kernels
        self.workers = min(8, self.numba.get_num_threads()) if workers is None else workers
        if self.workers > self.numba.config.NUMBA_NUM_THREADS:
            raise ValueError("workers exceeds Numba's configured maximum thread count")
        self.free = free
        self.shape = prob.shape
        self.p = np.ascontiguousarray(prob[free])
        ids = np.full(free.shape, -1, dtype=np.int64)
        ids[free] = np.arange(self.p.shape[0])
        padded = np.pad(ids, 1, constant_values=-2)
        self.neighbors = np.column_stack([
            padded[:-2, 1:-1][free], padded[2:, 1:-1][free],
            padded[1:-1, :-2][free], padded[1:-1, 2:][free],
        ])
        self.sx = np.empty_like(self.p)
        self.sy = np.empty_like(self.p)
        self.sa = np.empty_like(self.p)
        self.stage = np.empty_like(self.p)
        self.next = np.empty_like(self.p)
        self.minima = np.empty(self.p.shape[0], dtype=np.float64)
        self.cx, self.cy = vx/dx, vy/dx
        self.rx, self.ry = rx, ry
        self.diffusion = diffusion/dx**2
        self.angular_diffusion = d_r/dphi**2
        self.angular_speed = omega/dphi
        self.spatial_active = bool(np.any(vx) or np.any(vy))
        self.minimum = float(prob.min())

    def advance(self, count, h):
        previous_threads = self.numba.get_num_threads()
        self.numba.set_num_threads(self.workers)
        try:
            for _ in range(count):
                for finish in (False, True):
                    p = self.stage if finish else self.p
                    out = self.next if finish else self.stage
                    self.slopes(p, self.neighbors, self.sx, self.sy, self.sa,
                        self.spatial_active, self.angular_speed != 0.0)
                    self.update(p, self.p, self.neighbors, self.sx, self.sy,
                        self.sa, self.cx, self.cy, self.rx, self.ry, self.diffusion,
                        self.angular_diffusion, self.angular_speed, h, finish,
                        out, self.minima)
                    self.minimum = min(self.minimum, float(self.minima.min()))
                self.p, self.next = self.next, self.p
        finally:
            self.numba.set_num_threads(previous_threads)

    def full_probability(self):
        full = np.zeros(self.shape, dtype=self.p.dtype)
        full[self.free] = self.p
        return full


def _integer(name: str, value: object, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def _finite(name: str, value: object, *, minimum: float | None = None,
            positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0:
        raise ValueError(f"{name} must be > 0")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def euler_cabp_coefficients(
    q_p: float, q_xi: float, *, nu: float = 1.0, ell: float = 1.0,
) -> dict[str, float]:
    """Return the original proposed field-level parameter map.

    Dr=nu*(1-q_p), omega=nu*q_xi, v0=nu*ell, D=nu*ell**2/4.
    This is an assumed map, not a derivation from unspecified lattice rules.
    If q_p+i*q_xi is the first angular moment of a Poisson jump kernel,
    the angular map matches its first harmonic; higher harmonics generally
    differ. It is not the exact map for a fixed-time-step angular chain.
    Constraints on the original transition probabilities remain the caller's
    responsibility. q_p <= 1 is required to avoid negative Dr.
    """
    q_p = _finite("q_p", q_p)
    q_xi = _finite("q_xi", q_xi)
    nu = _finite("nu", nu, minimum=0.0)
    ell = _finite("ell", ell, positive=True)
    if q_p > 1:
        raise ValueError("q_p must be <= 1 so D_r is nonnegative")
    result = {
        "v0": nu * ell, "D": nu * ell * ell / 4.0,
        "D_r": nu * (1.0 - q_p), "omega": nu * q_xi,
        "nu": nu, "ell": ell, "q_p": q_p, "q_xi": q_xi,
    }
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError("coefficient map overflowed; rescale parameters")
    return result


def correlation_cabp_coefficients(
    q_p: float, q_xi: float, *, nu: float = 1.0, ell: float = 1.0,
    delta_t: Optional[float] = None,
) -> dict[str, float]:
    """Match the discrete-time aCRW's first angular correlation (Figure 5).

    lambda = q_p + i*q_xi, Dr = -log(abs(lambda))/delta_t,
    omega = arg(lambda)/delta_t, using the principal argument. Thus
    exp((-Dr+i*omega)*n*delta_t) = lambda**n at integer lags n.
    Positive q_xi is counterclockwise for the (right, up, left, down) coin.

    delta_t defaults to 1/nu. As in Figure 5's correlation branch, retain
    v0=nu*ell and D=nu*ell**2/4. Matching the first angular harmonic alone
    does not imply identical spatial distributions or higher harmonics.
    lambda=0 has no finite matching Dr and is rejected explicitly.
    """
    q_p = _finite("q_p", q_p)
    q_xi = _finite("q_xi", q_xi)
    nu = _finite("nu", nu, positive=True)
    ell = _finite("ell", ell, positive=True)
    if abs(q_p) + abs(q_xi) > 1.0:
        raise ValueError("|q_p| + |q_xi| must be <= 1 for the aCRW coin")
    delta_t = _finite("delta_t", 1.0 / nu if delta_t is None else delta_t, positive=True)
    radius = abs(complex(q_p, q_xi))
    if radius == 0.0:
        raise ValueError("lambda=0 has no finite correlation-matched D_r")
    angle = float(np.angle(complex(q_p, q_xi)))
    result = {
        "v0": nu * ell, "D": nu * ell * ell / 4.0,
        "D_r": max(0.0, -math.log(radius) / delta_t),
        "omega": angle / delta_t,
        "nu": nu, "ell": ell, "q_p": q_p, "q_xi": q_xi,
        "delta_t": delta_t, "abs_lambda": radius, "arg_lambda": angle,
    }
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError("correlation coefficient map overflowed; rescale parameters")
    return result


def square_defect_mask(
    n: int, *, offset: int = 3, period: int = 8, block: int = 4,
) -> np.ndarray:
    """Axis-aligned blocks anchored at offset, spaced by period.

    Every complete block fitting with at least offset cells of margin on both
    sides is included, even in an incomplete final period. offset specifies
    the first block's position; the array is not automatically recentered.
    True means a defect. Defaults give 4x4 blocks separated by four cells.
    """
    n = _integer("n", n, 1)
    offset = _integer("offset", offset)
    period = _integer("period", period, 1)
    block = _integer("block", block, 1)
    if block > period:
        raise ValueError("block must be <= period")
    on_block = np.zeros(n, dtype=bool)
    for start in range(offset, n - offset - block + 1, period):
        on_block[start:start + block] = True
    return on_block[:, None] & on_block[None, :]


def _minmod(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    same_sign = ((left > 0) & (right > 0)) | ((left < 0) & (right < 0))
    return np.where(same_sign, np.sign(left) * np.minimum(abs(left), abs(right)), 0.0)


def _axis_rhs(
    prob: np.ndarray, free: np.ndarray, velocity: np.ndarray,
    reflection: np.ndarray, dx: float, diffusion: float,
    *, return_faces: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Axis-0 probability rate; velocity has one entry per angular bin.

    Returns optional angle-summed face currents, oriented in +axis direction.
    Defect faces have zero total current but transfer angular-bin mass.
    """
    rate = np.zeros_like(prob)
    slope = np.zeros_like(prob)
    if prob.shape[0] >= 3:
        stencil = (free[:-2] & free[1:-1] & free[2:])[..., None]
        slope[1:-1] = np.where(stencil, _minmod(
            prob[1:-1] - prob[:-2], prob[2:] - prob[1:-1]), 0.0)

    speed = velocity / dx
    diffusivity = diffusion / dx**2
    both = free[:-1] & free[1:]
    left_state = prob[:-1] + 0.5 * slope[:-1]
    right_state = prob[1:] - 0.5 * slope[1:]
    face = np.where(speed >= 0, speed * left_state, speed * right_state)
    face += diffusivity * (prob[:-1] - prob[1:])
    face = np.where(both[..., None], face, 0.0)
    rate[:-1] -= face
    rate[1:] += face

    # Reflection is an angular-bin transfer in the free cell at the wall.
    hit_right = (free[:-1] & ~free[1:])[..., None]
    outgoing = np.where(hit_right, np.maximum(speed, 0.0) * prob[:-1], 0.0)
    rate[:-1] += outgoing[..., reflection] - outgoing
    hit_left = (free[1:] & ~free[:-1])[..., None]
    outgoing = np.where(hit_left, np.maximum(-speed, 0.0) * prob[1:], 0.0)
    rate[1:] += outgoing[..., reflection] - outgoing

    # The physical boundary is dx/2 from the center: factor TWO for diffusion.
    # Both terms also apply when the axis has only one cell.
    loss_left = np.where(free[0, :, None],
        (np.maximum(-speed, 0.0) + 2.0 * diffusivity) * prob[0], 0.0)
    loss_right = np.where(free[-1, :, None],
        (np.maximum(speed, 0.0) + 2.0 * diffusivity) * prob[-1], 0.0)
    rate[0] -= loss_left
    rate[-1] -= loss_right
    if not return_faces:
        return rate
    faces = np.zeros((prob.shape[0] + 1, prob.shape[1]), dtype=prob.dtype)
    faces[1:-1] = face.sum(axis=-1)
    faces[0] = -loss_left.sum(axis=-1)
    faces[-1] = loss_right.sum(axis=-1)
    return rate, faces


def _angular_rhs(prob: np.ndarray, dphi: float, d_r: float, omega: float) -> np.ndarray:
    """Periodic MUSCL drift plus centered diffusion, in probability-mass units."""
    right = np.roll(prob, -1, axis=-1)
    slope = _minmod(prob - np.roll(prob, 1, axis=-1), right - prob)
    if omega >= 0:
        face = (omega / dphi) * (prob + 0.5 * slope)
    else:
        face = (omega / dphi) * (right - 0.5 * np.roll(slope, -1, axis=-1))
    face += (d_r / dphi**2) * (prob - right)
    return np.roll(face, 1, axis=-1) - face


def _diffusion_degree(free: np.ndarray) -> int:
    """Maximum outgoing diffusion weight, including doubled exterior faces."""
    degree = np.zeros(free.shape, dtype=np.int16)
    pairs = free[:-1] & free[1:]
    degree[:-1] += pairs
    degree[1:] += pairs
    pairs = free[:, :-1] & free[:, 1:]
    degree[:, :-1] += pairs
    degree[:, 1:] += pairs
    degree[0] += 2 * free[0]
    degree[-1] += 2 * free[-1]
    degree[:, 0] += 2 * free[:, 0]
    degree[:, -1] += 2 * free[:, -1]
    return int(degree.max())


def _bin_to_lattice(prob: np.ndarray, lattice_n: int, refine: int) -> np.ndarray:
    return prob.sum(axis=-1).reshape(lattice_n, refine, lattice_n, refine).sum(axis=(1, 3))


def _active_flux_to_lattice(
    prob: np.ndarray, phi: np.ndarray, v0: float, lattice_n: int, refine: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cell-integrated active moment only; excludes the diffusive current.

    Units: probability*length/time. This is not a face-crossing rate or a
    density. Divide by ell**2 for a coarse cell-averaged active current density.
    """
    shape = (lattice_n, refine, lattice_n, refine)
    jx = (v0 * np.sum(prob * np.cos(phi), axis=-1)).reshape(shape).sum(axis=(1, 3))
    jy = (v0 * np.sum(prob * np.sin(phi), axis=-1)).reshape(shape).sum(axis=(1, 3))
    return jx, jy


def _initial_angular_weights(
    n_phi: int, initial_orientation: str | float | Sequence[float] = "uniform",
) -> np.ndarray:
    """Unit angular-bin masses for an isotropic or directed initial state.

    Named headings: right=0, up=pi/2, left=pi, down=3*pi/2. A numeric angle
    is in radians, counterclockwise from +x, with periodic wrapping. An angle
    between nodes is distributed over the two adjacent angular bins using
    nonnegative linear weights; it approximates a delta as n_phi increases.
    A length-n_phi vector of finite nonnegative weights is normalized to 1.
    """
    if isinstance(initial_orientation, str):
        name = initial_orientation.lower()
        if name == "uniform":
            return np.full(n_phi, 1.0 / n_phi)
        headings = {"right": 0.0, "up": 0.5 * np.pi,
                    "left": np.pi, "down": 1.5 * np.pi}
        if name not in headings:
            raise ValueError("initial_orientation must be 'uniform', 'right', 'up', "
                "'left', 'down', an angle in radians, or n_phi angular weights")
        initial_orientation = headings[name]
    if np.isscalar(initial_orientation):
        if isinstance(initial_orientation, (bool, np.bool_)):
            raise ValueError("initial_orientation must be a heading, not a boolean")
        angle = _finite("initial orientation angle", initial_orientation) % (2.0 * np.pi)
        coordinate = angle * n_phi / (2.0 * np.pi)
        # Cardinal / exact-node angles should populate exactly one bin.
        nearest = round(coordinate)
        if abs(coordinate - nearest) <= 16 * np.finfo(float).eps * n_phi:
            coordinate = float(nearest)
        index = math.floor(coordinate)
        fraction = coordinate - index
        weights = np.zeros(n_phi)
        weights[index % n_phi] = 1.0 - fraction
        weights[(index + 1) % n_phi] += fraction
        return weights
    raw = np.asarray(initial_orientation)
    if raw.shape != (n_phi,) or not np.isrealobj(raw):
        raise ValueError("initial angular weights must be a real vector of length n_phi")
    try:
        weights = raw.astype(float, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("initial angular weights must be finite nonnegative numbers") from exc
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("initial angular weights must be finite and nonnegative")
    peak = float(weights.max())
    if peak <= 0:
        raise ValueError("initial angular weights must have positive total mass")
    # Scaling before summing also handles very large finite supplied weights.
    weights /= peak
    weights /= weights.sum()
    return weights


def _point_source(
    lattice_n: int, refine: int, n_phi: int, origin: tuple[int, int],
    free: np.ndarray, dtype: np.dtype = np.dtype(np.float64),
    initial_orientation: str | float | Sequence[float] = "uniform",
) -> np.ndarray:
    """Legacy name: unit mass uniform over one lattice cell, with chosen heading."""
    try:
        x0, y0 = origin
    except (TypeError, ValueError) as exc:
        raise ValueError("origin must contain two integer lattice indices") from exc
    x0 = _integer("origin[0]", x0)
    y0 = _integer("origin[1]", y0)
    if x0 >= lattice_n or y0 >= lattice_n:
        raise ValueError("origin must lie inside the lattice")
    xs = slice(x0 * refine, (x0 + 1) * refine)
    ys = slice(y0 * refine, (y0 + 1) * refine)
    if not np.all(free[xs, ys]):
        raise ValueError(f"origin {origin} is not a fully free lattice cell")
    prob = np.zeros((lattice_n * refine, lattice_n * refine, n_phi), dtype=dtype)
    prob[xs, ys, :] = _initial_angular_weights(n_phi, initial_orientation) / refine**2
    return prob


def evolve_cabp_density(
    *, lattice_n: int, defect_lattice: np.ndarray, origin: tuple[int, int],
    v0: float, diffusion: float, d_r: float, omega: float,
    t_max: int, save_times: Sequence[int], ell: float = 1.0,
    refine: int = 2, n_phi: int = 32, dt: float = 0.2,
    lattice_rho: Optional[np.ndarray] = None, progress_every: int = 0,
    dtype: object = np.float64, backend: str = "auto",
    workers: Optional[int] = None, store_rho_series: bool = True,
    initial_orientation: str | float | Sequence[float] = "uniform",
    record_moments: bool = False,
) -> dict[str, object]:
    """Evolve unconditional cell probabilities and record integer times.

    dt is the requested macro step and must divide one unit of physical time.
    Smaller automatic internal steps enforce the combined positivity bound.
    No renormalization is performed: sum(rho) is the survival probability.
    rho/ell**2 is the coarse probability density. lattice_rho, if supplied,
    must also contain unconditional cell probabilities, not normalized
    survivor densities, and have shape (t_max+1, lattice_n, lattice_n).
    Tiny roundoff-level negatives in comparison data are accepted unchanged
    with a warning; larger per-entry or accumulated negative mass is rejected.

    flux_snapshots: active cell-integrated moments (original behavior).
    face_flux_snapshots: (Fx, Fy), total numerical coarse face currents with
        shapes (N+1,N), (N,N+1), positive along +x/+y, probability/time.
        These include diffusion and enforce zero net flux into defects.
    minimum_probability: minimum angular/spatial bin mass observed at RK
        stages, useful for checking positivity (not a density).

    float64 is the default; float32 is available to reduce memory consumption.
    Refinement studies in dx, dphi, and internal_dt remain necessary.

    backend: "auto" (Numba if importable, otherwise NumPy), "numba", or "numpy".
    workers: Numba threads, default at most 8; try 1, 2, 4, and 8 on your CPU.
        The previous thread setting is restored after every integer-time advance.
    store_rho_series=False: return rho_series=None; still record survival, L1,
        and selected snapshots. This reduces output memory, not solver work.
    Output backend reports the actual backend selected. Initial compilation
        costs several seconds on the first Numba call, then kernels are cached.

    initial_orientation: "uniform" (default), "right", "up", "left", "down",
        a finite angle in radians counterclockwise from +x, or a vector of
        n_phi nonnegative angular weights (normalized internally). Angles
        between angular nodes use the two neighboring bins. Named cardinal
        headings are exact nodes when n_phi is divisible by 4. This sets the
        initial heading only; omega and d_r continue to rotate and randomize it.
    record_moments: record unconditional spatial and angular moments at each
        integer time, using the internal spatial mesh. Spatial coordinates are
        relative to the center of the initial coarse cell. The initial cell
        has finite width; subtract its initial MSD for a displacement benchmark.
        These diagnostics do not change the evolution or normalize survivors.
    """
    lattice_n = _integer("lattice_n", lattice_n, 1)
    refine = _integer("refine", refine, 1)
    n_phi = _integer("n_phi", n_phi, 4)
    if n_phi % 2:
        raise ValueError("n_phi must be even for exact specular angle indices")
    t_max = _integer("t_max", t_max)
    progress_every = _integer("progress_every", progress_every)
    if backend not in ("auto", "numpy", "numba"):
        raise ValueError("backend must be 'auto', 'numpy', or 'numba'")
    if workers is not None:
        workers = _integer("workers", workers, 1)
    if not isinstance(store_rho_series, (bool, np.bool_)):
        raise ValueError("store_rho_series must be boolean")
    if not isinstance(record_moments, (bool, np.bool_)):
        raise ValueError("record_moments must be boolean")
    ell = _finite("ell", ell, positive=True)
    dt = _finite("dt", dt, positive=True)
    v0 = _finite("v0", v0, minimum=0.0)
    diffusion = _finite("diffusion", diffusion, minimum=0.0)
    d_r = _finite("d_r", d_r, minimum=0.0)
    omega = _finite("omega", omega)
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("dtype must be float32 or float64")
    reciprocal = 1.0 / dt
    if not math.isfinite(reciprocal):
        raise ValueError("dt is too small to represent its reciprocal")
    steps_per_unit = round(reciprocal)
    if steps_per_unit < 1 or abs(steps_per_unit * dt - 1.0) > 1e-12:
        raise ValueError("dt must divide 1 so the solution hits integer times")
    # Snap to the exact reciprocal to avoid accumulating accepted input roundoff.
    dt = 1.0 / steps_per_unit
    save_set = {_integer("save time", t) for t in save_times}
    if any(t > t_max for t in save_set):
        raise ValueError("save_times must lie between 0 and t_max")
    mask = np.asarray(defect_lattice)
    if mask.shape != (lattice_n, lattice_n):
        raise ValueError("defect_lattice must be (lattice_n, lattice_n)")
    if mask.dtype.kind not in "biuf" or not np.all(np.isin(mask, [0, 1])):
        raise ValueError("defect_lattice must contain only boolean or 0/1 values")
    free = np.repeat(np.repeat(~mask.astype(bool), refine, axis=0), refine, axis=1)
    prob = _point_source(lattice_n, refine, n_phi, origin, free, dtype, initial_orientation)
    lattice_rho_minimum = None
    lattice_rho_negative_mass_max = 0.0
    if lattice_rho is not None:
        lattice_rho = np.asarray(lattice_rho)
        if lattice_rho.shape != (t_max + 1, lattice_n, lattice_n):
            raise ValueError("lattice_rho must have shape (t_max+1, N, N)")
        if not np.isrealobj(lattice_rho) or not np.all(np.isfinite(lattice_rho)):
            raise ValueError("lattice_rho must contain finite real values")
        lattice_rho_minimum = float(lattice_rho.min())
        if lattice_rho_minimum < 0:
            # Comparison data never drive the PDE. Accept only tiny arithmetic
            # residuals, checking both each entry and the total negative mass
            # per frame; keep the supplied values unchanged when computing L1.
            eps = (np.finfo(lattice_rho.dtype).eps
                   if lattice_rho.dtype.kind == "f" else np.finfo(float).eps)
            eps = min(eps, np.finfo(np.float32).eps)
            tiny = np.finfo(float).tiny
            for frame in lattice_rho:
                frame_min = float(frame.min())
                if frame_min >= 0:
                    continue
                negative_mass = float(-frame[frame < 0].sum(dtype=np.float64))
                lattice_rho_negative_mass_max = max(lattice_rho_negative_mass_max, negative_mass)
                entry_tol = 64 * eps * max(float(np.max(np.abs(frame))), tiny)
                mass_tol = 64 * eps * max(float(np.abs(frame).sum(dtype=np.float64)), tiny)
                if frame_min < -entry_tol or negative_mass > mass_tol:
                    raise ValueError("lattice_rho has negative probabilities beyond roundoff: "
                        f"minimum={frame_min:.6g}, entry tolerance={entry_tol:.6g}, "
                        f"negative mass={negative_mass:.6g}, mass tolerance={mass_tol:.6g}. "
                        "Check the lattice transition weights and evolution.")
            warnings.warn("lattice_rho contains roundoff-level negative entries "
                f"(minimum {lattice_rho_minimum:.3g}, maximum negative mass per frame "
                f"{lattice_rho_negative_mass_max:.3g}); accepted unchanged for L1 comparison. "
                "Check how the lattice transition weights are constructed.",
                RuntimeWarning, stacklevel=2)

    dx = ell / refine
    dx_squared = dx * dx
    if not math.isfinite(dx_squared) or dx_squared == 0:
        raise ValueError("squared spatial spacing is not representable; rescale ell/refine")
    dphi = 2.0 * np.pi / n_phi
    phi = dphi * np.arange(n_phi)
    ux, uy = np.cos(phi), np.sin(phi)
    # Remove floating-point remnants of exactly tangential cardinal headings.
    ux[np.abs(ux) < 8 * np.finfo(float).eps] = 0.0
    uy[np.abs(uy) < 8 * np.finfo(float).eps] = 0.0
    vx, vy = v0 * ux, v0 * uy
    reflect_x = (n_phi // 2 - np.arange(n_phi)) % n_phi
    reflect_y = (-np.arange(n_phi)) % n_phi

    # For nonnegative states, a minmod reconstructed outgoing value is <=1.5*p.
    # Diffusion's exact diagonal includes weight 2 on each exterior face.
    loss_bound = (
        1.5 * v0 * float(np.max(abs(ux) + abs(uy))) / dx
        + diffusion * _diffusion_degree(free) / dx_squared
        + 1.5 * abs(omega) / dphi + 2.0 * d_r / dphi**2
    )
    if not math.isfinite(loss_bound):
        raise ValueError("discrete rates overflowed; rescale parameters")
    requested_substeps = dt * loss_bound / 0.9
    if not math.isfinite(requested_substeps):
        raise ValueError("required substep count overflowed; rescale parameters")
    substeps = max(1, math.ceil(requested_substeps))
    internal_dt = dt / substeps

    kernels = None
    if backend != "numpy":
        try:
            kernels = _load_numba()
        except ImportError as exc:
            if backend == "numba":
                raise ImportError("Numba acceleration needs a compatible NumPy/Numba installation. "
                    "Install both together in your Python environment, or use backend='numpy'.") from exc
            warnings.warn(f"Numba acceleration unavailable ({exc}); using the slower NumPy backend. "
                "Install compatible NumPy/Numba versions to enable acceleration.", RuntimeWarning, stacklevel=2)
    actual_backend = "numba" if kernels is not None else "numpy"
    stepper = None if kernels is None else _CompiledStepper(
        prob, free, vx, vy, reflect_x, reflect_y, dx, dphi,
        diffusion, d_r, omega, kernels, workers)

    def rhs(p: np.ndarray) -> np.ndarray:
        rate = _axis_rhs(p, free, vx, reflect_x, dx, diffusion)
        rate += np.swapaxes(_axis_rhs(np.swapaxes(p, 0, 1), free.T,
            vy, reflect_y, dx, diffusion), 0, 1)
        if d_r or omega:
            rate += _angular_rhs(p, dphi, d_r, omega)
        return rate

    snapshots: dict[int, np.ndarray] = {}
    flux_snapshots: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    face_flux_snapshots: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    survival = np.empty(t_max + 1, dtype=float)
    l1 = np.full(t_max + 1, np.nan, dtype=float)
    rho_series = (np.empty((t_max + 1, lattice_n, lattice_n), dtype=dtype)
                  if store_rho_series else None)
    minimum_probability = float(prob.min())

    moment_series = None
    if record_moments:
        moment_series = {key: np.empty(t_max + 1, dtype=float) for key in (
            "mean_x", "mean_y", "msd", "orientation_x", "orientation_y",
            "harmonic2_real", "harmonic2_imag")}
        moment_x = (np.arange(lattice_n * refine) + 0.5) * dx - (origin[0] + 0.5) * ell
        moment_y = (np.arange(lattice_n * refine) + 0.5) * dx - (origin[1] + 0.5) * ell

    def record(step: int) -> None:
        rho = _bin_to_lattice(prob, lattice_n, refine)
        survival[step] = rho.sum(dtype=np.float64)
        if moment_series is not None:
            spatial = prob.sum(axis=2, dtype=np.float64)
            px, py = spatial.sum(axis=1), spatial.sum(axis=0)
            angular = prob.sum(axis=(0, 1), dtype=np.float64)
            moment_series["mean_x"][step] = px @ moment_x
            moment_series["mean_y"][step] = py @ moment_y
            moment_series["msd"][step] = px @ (moment_x**2) + py @ (moment_y**2)
            moment_series["orientation_x"][step] = angular @ ux
            moment_series["orientation_y"][step] = angular @ uy
            moment_series["harmonic2_real"][step] = angular @ np.cos(2 * phi)
            moment_series["harmonic2_imag"][step] = angular @ np.sin(2 * phi)
        if rho_series is not None:
            rho_series[step] = rho
        if lattice_rho is not None:
            l1[step] = np.abs(rho - lattice_rho[step]).sum(dtype=np.float64)
        if step in save_set:
            snapshots[step] = rho
            flux_snapshots[step] = _active_flux_to_lattice(prob, phi, v0, lattice_n, refine)
            _, fx = _axis_rhs(prob, free, vx, reflect_x, dx, diffusion, return_faces=True)
            _, fy = _axis_rhs(np.swapaxes(prob, 0, 1), free.T,
                vy, reflect_y, dx, diffusion, return_faces=True)
            coarse_x = fx[::refine].reshape(lattice_n + 1, lattice_n, refine).sum(axis=-1)
            coarse_y = fy[::refine].reshape(lattice_n + 1, lattice_n, refine).sum(axis=-1).T
            face_flux_snapshots[step] = coarse_x, coarse_y

    record(0)
    for step in range(1, t_max + 1):
        if stepper is not None:
            stepper.advance(steps_per_unit * substeps, internal_dt)
            prob = stepper.full_probability()
            minimum_probability = stepper.minimum
        else:
            for _ in range(steps_per_unit * substeps):
                stage = prob + internal_dt * rhs(prob)
                advanced = stage + internal_dt * rhs(stage)
                # SSP RK2 is a convex combination of two positivity-safe Euler steps.
                prob = 0.5 * prob + 0.5 * advanced
                minimum_probability = min(minimum_probability, float(stage.min()),
                    float(advanced.min()), float(prob.min()))
        if not np.all(np.isfinite(prob)):
            raise FloatingPointError("non-finite probabilities during evolution")
        if minimum_probability < -100 * np.finfo(dtype).eps:
            raise FloatingPointError("negative probabilities exceed roundoff tolerance")
        record(step)
        if progress_every and step % progress_every == 0:
            print(f"  cABP t={step}/{t_max}  S={survival[step]:.6f}", flush=True)

    return {
        "snapshots": snapshots, "flux_snapshots": flux_snapshots,
        "face_flux_snapshots": face_flux_snapshots,
        "survival": survival, "rho_series": rho_series, "l1": l1,
        "moment_series": moment_series,
        "dx": dx, "dt": dt, "n_phi": n_phi, "refine": refine,
        "cfl": v0 * dt / dx, "internal_dt": internal_dt,
        "internal_cfl": v0 * internal_dt / dx,
        "substeps_per_dt": substeps, "minimum_probability": minimum_probability,
        "backend": actual_backend,
        "workers": stepper.workers if stepper is not None else None,
        "lattice_rho_minimum": lattice_rho_minimum,
        "lattice_rho_negative_mass_max": lattice_rho_negative_mass_max,
    }
