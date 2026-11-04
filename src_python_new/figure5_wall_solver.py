"""Orientation-resolved finite-volume solver for manuscript Figure 5.

Used by figure_scripts/figure5_cabp_vacf_msd.py. The driver supplies the
manuscript parameters, square box and near-wall source explicitly. Both models start with uniform orientations and the same Gaussian
spatial density, whose nominal radius containing 95% of mass is five physical
unit cells. Gaussian masses are integrated exactly over native cells; the source
is conditioned once to remain inside the box. No radial cutoff. Specular walls
are the default. Correlation-matched angular rates, v0=1, no independent
translational diffusion. No project scripts or input data are needed.
Requires NumPy, SciPy, Matplotlib; Numba is recommended for the full run.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import math
import os
import time
import warnings
from pathlib import Path
from typing import Sequence
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
from scipy import fft
from scipy.special import ndtr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ========================= EDIT THESE SETTINGS =========================
PLOT_TIMES = (50, 100, 200, 500, 1000)
BOX = 151                      # Physical width in common unit cells.
ORIGIN = (75, 10)               # Gaussian center=(75.5,10.5), as in Figure 15.
SOURCE_RADIUS95 = 5.0           # Physical radius containing 95% before domain conditioning.
INITIAL_ORIENTATION = "uniform"
# CHANGE THIS LINE to choose the boundary condition (or use --wall-rule):
# "figure2": blocked hops stay and receive the Figure 2 turning coin.
# "specular": mirror the heading at the wall in BOTH models.
# "stay": blocked motion stays without a wall-induced change of heading.
# Each choice saves separate caches and plots; switching preserves other runs.
WALL_RULE = "specular"
WEAK_TAU = 1.
STRONG_TAU = 1.
WEAK_D_R = -math.log(abs(.9+.1j))
WEAK_OMEGA = math.atan2(.1,.9)
STRONG_D_R = -math.log(abs(.6+.4j))
STRONG_OMEGA = math.atan2(.4,.6)
REFINE = 2                     # Continuum dx=1/REFINE.
N_PHI = 64                     # Multiple of four; continuous-angle approximation.
DT = .2                        # Strang angular / SSP RK2 spatial splitting.
WORKERS = 8                    # Threads per case. Two cases can run concurrently.
PARALLEL_CASES = 2
BACKEND = "auto"               # auto / numba / numpy.
PROBABILITY = .9
BIN_WIDTH = 2                  # Target width; equal bins cover the full physical box.
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "Figures -ACRW/Fig5 - Composite mapping comparison/wall_simulations"
# ======================================================================
SHIFTS = np.array([[1,0],[0,1],[-1,0],[0,-1]],dtype=np.int64)
prange = range
_KERNELS = None


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

def gaussian_spatial_mass(config,factor):
    """Exact Gaussian cell integrals, conditioned once on the physical box.

    The nominal center is (origin[0]+0.5, origin[1]+0.5), matching Figure 15.
    All meshes integrate the same continuous density. No radial cutoff or
    center-value sampling is used; the only truncation is the physical domain.
    """
    box=config['box'];sigma=config['source_sigma']
    edges=np.arange(box*factor+1,dtype=float)/factor
    marginals=[]
    for origin in config['origin']:
        z=(edges-(origin+.5))/sigma
        # Survival-function differences avoid cancellation in the right tail.
        weights=np.where(z[:-1]>=0,ndtr(-z[:-1])-ndtr(-z[1:]),ndtr(z[1:])-ndtr(z[:-1]))
        if np.any(weights<0) or not np.all(np.isfinite(weights)):
            raise FloatingPointError('Invalid Gaussian source integral')
        retained=float(weights.sum())
        if retained<=0:raise ValueError('Gaussian source has no mass in the box')
        marginals.append(weights/retained)
    return marginals[0][:,None]*marginals[1][None,:]


def source_description(config):
    """Continuous source parameters and analytical domain-conditioning effects."""
    box=config['box'];sigma=config['source_sigma']
    center=[v+.5 for v in config['origin']]
    retained=[];means=[];variances=[]
    for mu in center:
        a,b=-mu/sigma,(box-mu)/sigma
        z=float(ndtr(b)-ndtr(a));retained.append(z)
        pa,pb=math.exp(-.5*a*a)/math.sqrt(2*math.pi),math.exp(-.5*b*b)/math.sqrt(2*math.pi)
        shift=(pa-pb)/z
        means.append(mu+sigma*shift)
        variances.append(sigma*sigma*(1+(a*pa-b*pb)/z-shift*shift))
    radius=sigma*math.sqrt(2*math.log(20.))
    total=math.prod(retained)
    inside=all(radius<=mu<=box-radius for mu in center)
    return {'type':'isotropic Gaussian conditioned on the physical box',
        'nominal_center':center,'sigma_per_coordinate':sigma,'nominal_radius95':radius,
        'nominal_radius90':sigma*math.sqrt(2*math.log(10.)),
        'unconditioned_mass_retained_inside_box':total,
        'unconditioned_mass_outside_box':1.-total,
        'continuous_conditioned_mean':means,'continuous_conditioned_variance':variances,
        'conditioned_mass_within_nominal_radius95':.95/total if inside else None,
        'cell_initialization':'exact separable Gaussian CDF integrals on each native mesh',
        'angular_initialization':config['orientation']}


def integer_ratio(value, unit, label):
    ratio = value / unit
    if not np.isfinite(ratio) or not np.isclose(ratio, round(ratio), rtol=0, atol=1e-9):
        raise ValueError(f'{label}: {value} must be an integer multiple of {unit}')
    return int(round(ratio))

def mapped_coin(dr, omega, tau):
    lam = np.exp(complex(-dr, omega) * tau)
    if abs(lam.real) + abs(lam.imag) > 1 + 1e-13:
        raise ValueError('This correlation match is not a valid four-heading coin: '
                         '|q_p| + |q_c| > 1. Change Dr, omega, or tau.')
    qp, qc = float(lam.real), float(lam.imag)
    scale = abs(qp) + abs(qc)
    return (qp/scale, qc/scale) if scale > 1 else (qp, qc)

def make_coin(qp, qc):
    """Column-stochastic coin; positive qc turns right -> up -> left -> down."""
    if not np.isfinite(qp + qc) or abs(qp) + abs(qc) > 1:
        raise ValueError('The coin must satisfy |q_p| + |q_c| <= 1')
    eye = np.eye(4)
    return ((1 - (abs(qp) + abs(qc))) * np.full((4, 4), .25)
            + max(qp, 0) * eye + max(-qp, 0) * np.roll(eye, 2, axis=0)
            + max(qc, 0) * np.roll(eye, 1, axis=0)
            + max(-qc, 0) * np.roll(eye, -1, axis=0))


def _slopes(p, neighbors, sx, sy):
    for node in prange(p.shape[0]):
        left,right,down,up = neighbors[node]
        for a in range(p.shape[1]):
            q = p[node,a]
            x,y = 0.,0.
            if left>=0 and right>=0:
                dl,dr = q-p[left,a],p[right,a]-q
                if dl>0 and dr>0:x=min(dl,dr)
                elif dl<0 and dr<0:x=-min(-dl,-dr)
            if down>=0 and up>=0:
                dl,dr = q-p[down,a],p[up,a]-q
                if dl>0 and dr>0:y=min(dl,dr)
                elif dl<0 and dr<0:y=-min(-dl,-dr)
            sx[node,a],sy[node,a] = x,y


def _space_update(p,base,neighbors,sx,sy,cx,cy,rx,ry,wall_rule,qp,qc,wall_scale,h,finish,out):
    na = p.shape[1]
    for node in prange(p.shape[0]):
        # Total outward mass flux for the uniform component of the wall coin.
        totals = np.zeros(4)
        if wall_rule==0:
            for side in range(4):
                if neighbors[node,side]<0:
                    velocity = cx if side<2 else cy
                    sign = -1. if side%2==0 else 1.
                    total = 0.
                    for a in range(na):
                        total += max(sign*velocity[a],0.)*p[node,a]
                    totals[side] = total
        for a in range(na):
            q = p[node,a]
            rate = 0.
            for axis in range(2):
                velocity = cx if axis==0 else cy
                slope = sx if axis==0 else sy
                c = velocity[a]
                reflected = rx[a] if axis==0 else ry[a]
                left,right = neighbors[node,2*axis],neighbors[node,2*axis+1]
                if left>=0:
                    state = p[left,a]+.5*slope[left,a] if c>=0 else q-.5*slope[node,a]
                    rate += c*state
                if right>=0:
                    state = q+.5*slope[node,a] if c>=0 else p[right,a]-.5*slope[right,a]
                    rate -= c*state
                for side in range(2):
                    neighbor = left if side==0 else right
                    if neighbor>=0:continue
                    sign = -1. if side==0 else 1.
                    outgoing = max(sign*c,0.)*q
                    if wall_rule==1: # Specular face reflection.
                        incoming = max(sign*velocity[reflected],0.)*p[node,reflected]
                        rate += incoming-outgoing
                    elif wall_rule==0: # Figure 2 coin acting on outward flux.
                        reverse = (a+na//2)%na
                        from_ccw = (a-na//4)%na
                        from_cw = (a+na//4)%na
                        incoming = (max(qp,0.)*outgoing
                            +max(-qp,0.)*max(sign*velocity[reverse],0.)*p[node,reverse]
                            +max(qc,0.)*max(sign*velocity[from_ccw],0.)*p[node,from_ccw]
                            +max(-qc,0.)*max(sign*velocity[from_cw],0.)*p[node,from_cw]
                            +(1-(abs(qp)+abs(qc)))*totals[2*axis+side]/na)
                        rate += wall_scale*(incoming-outgoing)
                    # Stay: zero normal flux without a wall angular transfer.
            advanced = q+h*rate
            out[node,a] = .5*base[node,a]+.5*advanced if finish else advanced


def _lattice_step(p,coin,wall_coin,out):
    n = p.shape[0]
    for x in prange(n):
        for y in range(n):
            for a in range(4):
                sx,sy = x-SHIFTS[a,0],y-SHIFTS[a,1]
                value = 0.
                if 0<=sx<n and 0<=sy<n:
                    for j in range(4):value += coin[a,j]*p[sx,sy,j]
                for d in range(4):
                    dx,dy = x+SHIFTS[d,0],y+SHIFTS[d,1]
                    if not (0<=dx<n and 0<=dy<n):
                        blocked = 0.
                        for j in range(4):blocked += coin[d,j]*p[x,y,j]
                        value += wall_coin[a,d]*blocked
                out[x,y,a] = value


def load_kernels(backend,workers):
    global prange,_KERNELS
    if backend=='numpy':return None
    try:
        import numba
    except ImportError:
        if backend=='numba':raise
        warnings.warn('Numba unavailable; reference loops are substantially slower',RuntimeWarning)
        return None
    if workers>numba.config.NUMBA_NUM_THREADS:
        raise ValueError('workers exceeds NUMBA_NUM_THREADS; lower --workers or adjust that environment setting')
    numba.set_num_threads(workers)
    prange = numba.prange
    if _KERNELS is None:
        _KERNELS = (numba.njit(parallel=True,cache=True,nogil=True)(_slopes),
                    numba.njit(parallel=True,cache=True,nogil=True)(_space_update),
                    numba.njit(parallel=True,cache=True,nogil=True)(_lattice_step))
    return _KERNELS


def wall_code(rule):
    return {'figure2':0,'specular':1,'stay':2}[rule]


def wall_matrix(coin,rule):
    if rule=='figure2':return coin.copy()
    return np.roll(np.eye(4),2,axis=0) if rule=='specular' else np.eye(4)


def angular_kernel(dr,omega,h,n_phi):
    """Positive sampled wrapped-Brownian transition, with drift +omega.

    Normalize the transition kernel (never the solution). Aliasing errors
    of its first harmonic are measured explicitly against exp((-Dr+i*w)h).
    """
    phi = 2*np.pi*np.arange(n_phi)/n_phi
    delta = (phi-omega*h+np.pi)%(2*np.pi)-np.pi
    weights = np.zeros(n_phi)
    for image in range(-3,4):
        weights += np.exp(-(delta+2*np.pi*image)**2/(4*dr*h))
    weights /= weights.sum()
    expected = np.exp(complex(-dr,omega)*h)
    error = float(abs(weights@np.exp(1j*phi)-expected))
    if error>1e-7:
        raise ValueError(f'Angular kernel underresolved ({error:g}); increase n-phi or choose a larger macro dt')
    return fft.rfft(weights),error


def continuum(config):
    box,refine,na = config['box'],config['refine'],config['n_phi']
    n = box*refine
    dx = 1./refine
    phi = 2*np.pi*np.arange(na)/na
    cx,cy = np.cos(phi)/dx,np.sin(phi)/dx
    cx[abs(cx)<1e-14]=0.;cy[abs(cy)<1e-14]=0.
    # Strang half angular step; spatial SSP RK2 is independently positive.
    steps_per_unit = integer_ratio(1.,config['dt'],'dt')
    qp,qc = mapped_coin(config['dr'],config['omega'],config['tau'])
    wall_scale = dx/config['tau']
    diagonal = max(qp,0.)+(1-(abs(qp)+abs(qc)))/na
    bound = 1.5*float(np.max(abs(cx)+abs(cy)))
    if config['wall_rule']=='figure2':
        bound = max(1.5,wall_scale*(1-diagonal))*float(np.max(abs(cx)+abs(cy)))
    substeps = max(1,math.ceil(config['dt']*bound/.9))
    h = config['dt']/substeps
    kernel,kernel_error = angular_kernel(config['dr'],config['omega'],.5*h,na)
    kernels = load_kernels(config['backend'],config['workers'])
    slopes,update = (_slopes,_space_update) if kernels is None else kernels[:2]
    ids = np.arange(n*n).reshape(n,n)
    padded = np.pad(ids,1,constant_values=-1)
    neighbors = np.column_stack([padded[:-2,1:-1].ravel(),padded[2:,1:-1].ravel(),
                                padded[1:-1,:-2].ravel(),padded[1:-1,2:].ravel()])
    spatial = gaussian_spatial_mass(config,refine)
    p = (spatial[:,:,None]*_initial_angular_weights(na,config['orientation'])).reshape(n*n,na)
    sx,sy,stage,next_p = (np.empty_like(p) for _ in range(4))
    rx = (na//2-np.arange(na))%na
    ry = (-np.arange(na))%na
    qp,qc = mapped_coin(config['dr'],config['omega'],config['tau'])
    maximum = max(config['times'])
    survival = np.empty(maximum+1)
    snapshots = {}
    removed_roundoff = 0.
    minimum_stage = 0.
    started = time.perf_counter()
    def record(t,prob):
        rho = prob.reshape(box,refine,box,refine,na).sum(axis=(1,3,4))
        survival[t] = rho.sum()
        if t in config['times']:snapshots[t]=rho
    def rotate(prob,transition):
        nonlocal removed_roundoff
        transformed = fft.irfft(fft.rfft(prob,axis=-1,workers=config['workers'])*transition,
                               n=na,axis=-1,workers=config['workers'])
        negative = transformed<0
        if np.any(negative):
            amount = float(-transformed[negative].sum())
            if amount>1e-11:
                raise FloatingPointError(f'Material negative angular probability: {amount:g}')
            removed_roundoff += amount
            transformed[negative] = 0.
        return np.ascontiguousarray(transformed)
    record(0,p)
    # R(h/2) S(h) R(h/2) steps share adjacent angular half steps.
    # Store spatial marginals directly after S: angular rotation preserves
    # the mass in each spatial cell. This halves the number of FFT pairs.
    per_unit = steps_per_unit*substeps
    p = rotate(p,kernel)
    for t in range(1,maximum+1):
        for j in range(per_unit):
            slopes(p,neighbors,sx,sy)
            update(p,p,neighbors,sx,sy,cx,cy,rx,ry,wall_code(config['wall_rule']),qp,qc,wall_scale,h,False,stage)
            slopes(stage,neighbors,sx,sy)
            update(stage,p,neighbors,sx,sy,cx,cy,rx,ry,wall_code(config['wall_rule']),qp,qc,wall_scale,h,True,next_p)
            minimum_stage = min(minimum_stage,float(stage.min()),float(next_p.min()))
            if j==per_unit-1:
                record(t,next_p)
            if t!=maximum or j!=per_unit-1:
                p = rotate(next_p,kernel*kernel)
        if minimum_stage < -1e-14 or not np.all(np.isfinite(next_p)):
            raise FloatingPointError('Nonfinite or materially negative probability in spatial evolution')
        if t%25==0 or t==maximum:
            print(f'  {config["label"]}: cABP t={t}/{maximum}, S={survival[t]:.10f}',flush=True)
    return {'snapshots':snapshots,'survival':survival,
        'solver':{'method':'Strang angular convolution / minmod SSP RK2 spatial FV',
                  'backend':'numba' if kernels else 'numpy','dx':dx,'internal_dt':h,
                  'angular_half_step_first_harmonic_error':kernel_error,
                  'minimum_spatial_stage_probability':minimum_stage,
                  'removed_angular_fft_roundoff_mass':removed_roundoff,
                  'wall_reorientation_attempt_rate':('abs(normal velocity)/(v0*tau), independent of FV dx' if config['wall_rule']=='figure2' else 'none; mirror reflection' if config['wall_rule']=='specular' else 'none; blocked motion'),
                  'elapsed_seconds':time.perf_counter()-started}}


def lattice(config):
    factor = integer_ratio(1.,config['tau'],'tau')
    n = config['box']*factor
    coin = make_coin(*mapped_coin(config['dr'],config['omega'],config['tau']))
    collision = wall_matrix(coin,config['wall_rule'])
    spatial = gaussian_spatial_mass(config,factor)
    p = spatial[:,:,None]*_initial_angular_weights(4,config['orientation'])
    out = np.empty_like(p)
    kernels = load_kernels(config['backend'],config['workers'])
    advance = _lattice_step if kernels is None else kernels[2]
    maximum = max(config['times'])
    survival = np.empty(maximum+1)
    snapshots = {}
    def record(t):
        rho = p.sum(axis=2).reshape(config['box'],factor,config['box'],factor).sum(axis=(1,3))
        survival[t]=rho.sum()
        if t in config['times']:snapshots[t]=rho
    record(0)
    started=time.perf_counter()
    for step in range(1,maximum*factor+1):
        advance(p,coin,collision,out)
        p,out = out,p
        if step%factor==0:
            t=step//factor
            record(t)
            if t%100==0 or t==maximum:
                print(f'  {config["label"]}: aCRW t={t}/{maximum}, S={survival[t]:.10f}',flush=True)
    return {'snapshots':snapshots,'survival':survival,'elapsed_seconds':time.perf_counter()-started}


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


def observed_density(rho,width):
    """Equal physical observation bins covering the full box conservatively.

    Target width two in a 151-wide box gives 76 bins of width 151/76,
    avoiding a half-width last bin that would amplify wall-layer density.
    Input unit-cell masses are integrated by exact interval overlaps.
    """
    n = rho.shape[0]
    edges = np.linspace(0.,float(n),math.ceil(n/width)+1)
    cells = np.arange(n,dtype=float)
    overlap = np.maximum(0.,np.minimum(edges[1:,None],cells[None,:]+1.)
                         -np.maximum(edges[:-1,None],cells[None,:]))
    mass = overlap@rho@overlap.T
    sizes = np.diff(edges)
    return mass,mass/(sizes[:,None]*sizes[None,:]),edges


def hdr_level(mass,density,probability):
    if not 0<probability<1 or not np.all(np.isfinite(mass)) or np.min(mass)<0:
        raise ValueError('HDR requires finite nonnegative masses and 0<probability<1')
    total = float(mass.sum())
    if total<=0:raise ValueError('No probability mass')
    order = np.argsort(density.ravel())[::-1]
    cumulative = np.cumsum(mass.ravel()[order])
    j = min(int(np.searchsorted(cumulative,probability*total-32*np.finfo(float).eps*total)),len(order)-1)
    level = float(density.ravel()[order[j]])
    region = density>=level
    return level,{'grid_probability_at_or_above_cutoff':float(mass[region].sum()/total),
                 'survival_probability':total,'threshold_tie_probability':float(mass[density==level].sum()/total)}


def _wall_segments(coordinates,values,level):
    """Intervals of a physical box face belonging to the HDR."""
    intervals=[]
    start=coordinates[0] if values[0]>=level else None
    for x0,x1,y0,y1 in zip(coordinates[:-1],coordinates[1:],values[:-1],values[1:]):
        if (y0>=level)!=(y1>=level):
            cross=x0+(level-y0)*(x1-x0)/(y1-y0)
            if y1>=level:start=cross
            else:
                intervals.append((start,cross));start=None
    if start is not None:intervals.append((start,coordinates[-1]))
    return intervals


def plot(cases,args):
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix',
                         'axes.labelsize':12,'legend.fontsize':9,'axes.titlesize':12})
    fig,axes=plt.subplots(1,2,figsize=(10.4,6.1),sharex=True,sharey=True)
    fig.subplots_adjust(left=.08,right=.985,bottom=.15,top=.79,wspace=.12)
    times=cases[0][1]['times'][1:]
    palette=['#453781','#2C728E','#22A884','#C58A20','#CD5A3C']
    if len(times)>5:palette=plt.get_cmap('viridis')(np.linspace(.15,.85,len(times)))
    records=[]
    for ax,(data,config) in zip(axes,cases):
        ox,oy=config['origin']
        for j,t in enumerate(times,1):
            for model,field,linestyle,alpha,line_width in (
                    ('aCRW',data['lattice_snapshots'][j],'-',.65,2.2),
                    ('cABP',data['cabp_snapshots'][j],'--',1.,1.8)):
                mass,density,edges=observed_density(field,args.bin_width)
                level,audit=hdr_level(mass,density,args.probability)
                # Include physical wall faces using the boundary-cell density.
                centers=.5*(edges[:-1]+edges[1:])
                x=np.r_[edges[0],centers,edges[-1]]-(ox+.5)
                y=np.r_[edges[0],centers,edges[-1]]
                padded=np.pad(density,1,mode='edge')
                plot_level=level
                if level>=padded.max():
                    lower=padded[padded<level]
                    if not lower.size:raise ValueError('HDR fills the entire domain; use a smaller probability')
                    plot_level=.5*(level+float(lower.max()))
                contour=ax.contour(x,y,padded.T,levels=[plot_level],colors=[palette[j-1]],
                                   linestyles=[linestyle],linewidths=[line_width],alpha=alpha)
                components=sum(len(s)>2 for s in contour.allsegs[0])
                wall_parts=0
                # An HDR touching a wall is bounded by that wall, not a
                # fictitious fall to zero beyond it. Close those pieces.
                for side,coordinate,values in (
                        ('left',y,padded[0,:]),('right',y,padded[-1,:]),
                        ('bottom',x,padded[:,0]),('top',x,padded[:,-1])):
                    for start,end in _wall_segments(coordinate,values,plot_level):
                        if side in ('left','right'):
                            xx=x[0] if side=='left' else x[-1]
                            ax.plot([xx,xx],[start,end],color=palette[j-1],ls=linestyle,lw=line_width,alpha=alpha,clip_on=False)
                        else:
                            yy=y[0] if side=='bottom' else y[-1]
                            ax.plot([start,end],[yy,yy],color=palette[j-1],ls=linestyle,lw=line_width,alpha=alpha,clip_on=False)
                        wall_parts+=1
                records.append({'case':config['label'],'time':t,'model':model,'density_threshold':level,
                                'plotted_level':plot_level,'interior_contour_components':components,
                                'physical_wall_segments':wall_parts,'physical_bin_width':float(edges[1]-edges[0]),**audit})
        box=config['box']
        ax.plot(0,oy+.5,'kx',ms=5,label='Gaussian center',zorder=20)
        ax.set(xlim=(-(ox+.5),box-(ox+.5)),ylim=(0,box),aspect='equal',
               xlabel=r'$x-x_0$',title=config['label']+'\n'+rf'$\tau={config["tau"]:g}$')
        ax.tick_params(direction='out')
    axes[0].set_ylabel('Height above bottom wall')
    fig.suptitle(f'Figure 17. {args.probability:.0%} probability contours from a Gaussian start',y=.98,fontsize=15)
    handles=[Line2D([0],[0],color=palette[i],lw=2,label=rf'$t={t}$') for i,t in enumerate(times)]
    handles += [Line2D([0],[0],color='.25',lw=2.2,alpha=.65,label='aCRW'),
                Line2D([0],[0],color='.25',lw=1.8,ls='--',label='cABP')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.93),ncol=min(7,len(handles)),frameon=False)
    wall_label={'figure2':'Reflecting walls with angular scattering','specular':'Specular reflecting walls','stay':'No-penetration walls'}[args.wall_rule]
    fig.text(.5,.025,f'{wall_label}; {args.orientation} headings; '+rf'$R_{{95}}={args.source_radius95:g}$, $y_0={args.origin[1]+.5:g}$.',ha='center',fontsize=10)
    stem='figure5_gaussian_wall_'+args.wall_rule
    for extension in ('png','pdf','svg'):
        fig.savefig(args.output_dir/f'{stem}.{extension}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    return stem,records


def fingerprint():
    tree=ast.parse(Path(__file__).read_text())
    excluded={'plot','main','fingerprint','_wall_segments','observed_density','hdr_level'}
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name not in excluded]
    return hashlib.sha256(ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False).encode()).hexdigest()


def validate(data,config):
    checks={}
    exact=gaussian_spatial_mass(config,1)
    for model in ('cabp','lattice'):
        values=data[model+'_snapshots']
        survival=data[model+'_survival']
        if not np.all(np.isfinite(values)) or np.min(values)<-1e-14:
            raise FloatingPointError(model+': invalid probabilities')
        source_error=float(abs(values[0]-exact).sum())
        if source_error>1e-12:raise FloatingPointError(model+': inconsistent Gaussian initial condition')
        mass_error=float(abs(survival-1).max())
        if mass_error>1e-8:
            raise FloatingPointError(f'{model}: reflecting boundary mass error {mass_error:g}')
        checks[model]={'minimum_snapshot_probability':float(values.min()),'maximum_mass_error':mass_error,'initial_Gaussian_unit_cell_L1_error':source_error}
    return checks


def run_case(config):
    output=Path(config['output_dir'])
    output.mkdir(parents=True,exist_ok=True)
    cache=output/f'figure5_density_{config["slug"]}_{config["wall_rule"]}.npz'
    signature=json.dumps({k:v for k,v in config.items() if k not in ('recompute','output_dir')},sort_keys=True)
    if cache.exists() and not config['recompute']:
        with np.load(cache,allow_pickle=False) as stored:
            if str(stored['configuration_json'].item())==signature:
                data={key:stored[key].copy() for key in stored.files}
                print('Reusing '+str(cache),flush=True)
                return data,config,validate(data,config)
    print(f'{config["label"]}: confined box={config["box"]}, tau={config["tau"]}, wall={config["wall_rule"]}',flush=True)
    start=time.perf_counter()
    cont=continuum(config)
    lat=lattice(config)
    times=config['times']
    data={'times':np.array(times),'cabp_snapshots':np.array([cont['snapshots'][t] for t in times]),
          'lattice_snapshots':np.array([lat['snapshots'][t] for t in times]),
          'cabp_survival':cont['survival'],'lattice_survival':lat['survival'],
          'initial_density_exact_unit_cells':gaussian_spatial_mass(config,1),
          'initial_condition_json':np.array(json.dumps(source_description(config))),
          'configuration_json':np.array(signature),'solver_json':np.array(json.dumps(cont['solver'])),
          'elapsed_seconds':np.array(time.perf_counter()-start),
          'simulation_script_sha256':np.array(hashlib.sha256(Path(__file__).read_bytes()).hexdigest())}
    checks=validate(data,config)
    np.savez_compressed(cache,**data)
    print(f'{config["label"]}: finished in {float(data["elapsed_seconds"]):.1f}s; checks={checks}',flush=True)
    return data,config,checks


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--times',type=int,nargs='+',default=list(PLOT_TIMES))
    parser.add_argument('--box',type=int,default=BOX)
    parser.add_argument('--origin',type=int,nargs=2,default=ORIGIN,help='Reference cell: Gaussian center is (x+0.5,y+0.5)')
    width=parser.add_mutually_exclusive_group()
    width.add_argument('--source-radius95',type=float,help='95-percent radius of the unbounded Gaussian; default SOURCE_RADIUS95')
    width.add_argument('--source-sigma',type=float,help='Override Gaussian standard deviation per coordinate')
    parser.add_argument('--wall-rule',choices=['figure2','specular','stay'],default=WALL_RULE,
        help='Wall rule for both models: figure2=extra turning coin, specular=mirror heading, stay=keep heading (default: %(default)s). Overrides WALL_RULE above.')
    parser.add_argument('--orientation',choices=['uniform','up','right','left','down'],default=INITIAL_ORIENTATION)
    parser.add_argument('--weak-tau',type=float,default=WEAK_TAU)
    parser.add_argument('--strong-tau',type=float,default=STRONG_TAU)
    parser.add_argument('--weak-dr',type=float,default=WEAK_D_R)
    parser.add_argument('--weak-omega',type=float,default=WEAK_OMEGA)
    parser.add_argument('--strong-dr',type=float,default=STRONG_D_R)
    parser.add_argument('--strong-omega',type=float,default=STRONG_OMEGA)
    parser.add_argument('--refine',type=int,default=REFINE)
    parser.add_argument('--n-phi',type=int,default=N_PHI)
    parser.add_argument('--dt',type=float,default=DT)
    parser.add_argument('--workers',type=int,default=WORKERS)
    parser.add_argument('--parallel-cases',type=int,choices=[1,2],default=PARALLEL_CASES)
    parser.add_argument('--backend',choices=['auto','numba','numpy'],default=BACKEND)
    parser.add_argument('--probability',type=float,default=PROBABILITY)
    parser.add_argument('--bin-width',type=int,default=BIN_WIDTH)
    parser.add_argument('--output-dir',type=Path,default=OUTPUT_DIR)
    parser.add_argument('--recompute',action='store_true')
    args=parser.parse_args(argv)
    if args.source_sigma is None:
        args.source_radius95=SOURCE_RADIUS95 if args.source_radius95 is None else args.source_radius95
        args.source_sigma=args.source_radius95/math.sqrt(2*math.log(20.))
    else:args.source_radius95=args.source_sigma*math.sqrt(2*math.log(20.))
    try:
        if not np.isfinite(args.source_sigma) or args.source_sigma<=0:raise ValueError('Gaussian width must be finite and positive')
        if args.box<4 or any(v<0 or v>=args.box for v in args.origin):raise ValueError('Source must lie inside box >=4')
        if not args.times or min(args.times)<1:raise ValueError('Times must be positive integers')
        if args.refine<1 or args.n_phi<8 or args.n_phi%4:raise ValueError('refine >=1; n-phi must be a multiple of four and >=8')
        if args.workers<1 or args.bin_width<1 or not 0<args.probability<1:raise ValueError('Invalid workers, bin width or probability')
        if not np.isfinite(args.dt) or args.dt<=0:raise ValueError('dt must be finite and positive')
        integer_ratio(1.,args.dt,'dt')
        for dr,omega,tau in ((args.weak_dr,args.weak_omega,args.weak_tau),(args.strong_dr,args.strong_omega,args.strong_tau)):
            if not all(np.isfinite([dr,omega,tau])) or dr<=0 or tau<=0:raise ValueError('Angular rates must be finite; Dr and tau positive')
            integer_ratio(1.,tau,'tau')
            mapped_coin(dr,omega,tau)
    except ValueError as exc:parser.error(str(exc))
    args.output_dir=args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    print(f'Wall rule: {args.wall_rule}; separate results in {args.output_dir}',flush=True)
    configs=[]
    for label,slug,dr,omega,tau in (
            ('Weak turning','weak_turning',args.weak_dr,args.weak_omega,args.weak_tau),
            ('Strong turning','strong_turning',args.strong_dr,args.strong_omega,args.strong_tau)):
        configs.append({'label':label,'slug':slug,'dr':dr,'omega':omega,'tau':tau,
            'box':args.box,'origin':list(args.origin),'source_sigma':args.source_sigma,
            'source_radius95':args.source_radius95,'orientation':args.orientation,'wall_rule':args.wall_rule,
            'times':sorted(set([0]+args.times)),'refine':args.refine,'n_phi':args.n_phi,'dt':args.dt,
            'workers':args.workers,'backend':args.backend,'output_dir':str(args.output_dir),
            'recompute':args.recompute,'schema':2,'physics_sha256':fingerprint()})
    if args.parallel_cases==2:
        with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:
            rows=list(pool.map(run_case,configs))
    else:rows=[run_case(c) for c in configs]
    stem,levels=plot([(data,cfg) for data,cfg,_ in rows],args)
    metadata={'figure':17,'panel_order':['Weak turning','Strong turning'],
        'times':configs[0]['times'],'wall_rule':args.wall_rule,'physical_box':args.box,'source_cell':list(args.origin),
        'initial_condition':source_description(configs[0]),
        'coefficient_convention':{'angular_mapping':'exact first-harmonic correlation match','v0':1.,'translational_diffusion':0.},
        'probability':args.probability,'target_bin_width':args.bin_width,'actual_bin_width':args.box/math.ceil(args.box/args.bin_width),'cases':[
            {'configuration':cfg,'validation':checks,'solver':json.loads(str(data['solver_json'].item())),
             'elapsed_seconds':float(data['elapsed_seconds']),'simulation_script_sha256':str(data['simulation_script_sha256'].item())}
            for data,cfg,checks in rows],'contours':levels,
        'wall_note':({'figure2':'Blocked lattice hops stay and receive the bulk coin again. cABP outward motion is blocked; the corresponding angular jump kernel acts on outward headings in boundary cells at physical rate abs(normal velocity)/(v0*tau), with continuous angular diffusion/drift also present. This is an explicit wall-clock analogue, not a uniquely derived continuum wall law.', 'specular':'Both models mirror the heading normal to the impacted face; probability remains inside the domain.', 'stay':'Outward motion is blocked without a wall reorientation; the bulk angular dynamics continue.'}[args.wall_rule]),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output_dir/f'{stem}_metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    print('Saved '+str(args.output_dir/stem),flush=True)
    return rows,metadata


if __name__=='__main__':
    main()
