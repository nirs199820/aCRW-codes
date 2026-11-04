"""Figure 6: Gaussian-start wall comparison, shown as density excess.

Standalone NumPy/Matplotlib script. Run with:
  python figure6_steady_state_reflective_boundaries.py
  python figure6_steady_state_reflective_boundaries.py --plot-only

The initial Gaussian has nominal sigma=1, is integrated over the central 5x5
unit cells, conditioned on those cells, and split equally over four headings.
Only specular and randomized walls are shown. Each square is a split-time
half-domain view: its upper spatial half is t=150 and lower half t=5000.
Each time half has one symmetric colour scale shared across the wall rules.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "Figures -ACRW/Fig6 - Specular and randomized walls"
N = 41
Q_P, Q_C = .6, .2
TIMES = (150, 5000)
ORIGIN = (20, 20)
GAUSSIAN_SIGMA = 1.
SOURCE_SIDE = 5
MODES = ("Specular", "Randomized")
PROFILE_COLORS = {"Specular": "#D21517", "Randomized": "#3F70AD"}
SHIFTS = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])


def make_coin(qp, qc):
    """Column stochastic; headings right, up, left, down; positive qc is CCW."""
    assert abs(qp)+abs(qc) <= 1
    eye = np.eye(4)
    return ((1-(abs(qp)+abs(qc)))*np.full((4, 4), .25)
            + max(qp, 0)*eye + max(-qp, 0)*np.roll(eye, 2, axis=0)
            + max(qc, 0)*np.roll(eye, 1, axis=0)
            + max(-qc, 0)*np.roll(eye, -1, axis=0))


def initial_state(sigma=GAUSSIAN_SIGMA):
    """Cell-integrated central Gaussian conditioned on the 5x5 source patch."""
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("Gaussian sigma must be positive and finite")
    radius = SOURCE_SIDE//2
    offsets = np.arange(-radius, radius+1)
    cdf = lambda z: .5*(1+math.erf(z/math.sqrt(2)))
    raw = np.array([cdf((k+.5)/sigma)-cdf((k-.5)/sigma) for k in offsets])
    retained_mass = float(raw.sum()**2)
    weights = raw/raw.sum()
    rho = np.zeros((N, N))
    rho[ORIGIN[0]-radius:ORIGIN[0]+radius+1,
        ORIGIN[1]-radius:ORIGIN[1]+radius+1] = weights[:, None]*weights[None, :]
    p = np.repeat(rho[:, :, None]/4, 4, axis=2)
    return p, retained_mass


def step(p, coin, mode):
    """Bulk coin then shift; blocked mass stays at the departure wall cell."""
    q = np.einsum("ij,xyj->xyi", coin, p)
    out = np.zeros_like(q)
    out[1:, :, 0] = q[:-1, :, 0]
    out[:, 1:, 1] = q[:, :-1, 1]
    out[:-1, :, 2] = q[1:, :, 2]
    out[:, :-1, 3] = q[:, 1:, 3]
    if mode == "Specular":
        # Reflect only the outward normal heading at each physical wall.
        out[-1, :, 2] += q[-1, :, 0]   # right wall: right -> left
        out[0, :, 0] += q[0, :, 2]     # left wall: left -> right
        out[:, -1, 3] += q[:, -1, 1]   # top wall: up -> down
        out[:, 0, 1] += q[:, 0, 3]     # bottom wall: down -> up
    else:
        blocked = np.zeros_like(q)
        blocked[-1, :, 0] = q[-1, :, 0]
        blocked[0, :, 2] = q[0, :, 2]
        blocked[:, -1, 1] = q[:, -1, 1]
        blocked[:, 0, 3] = q[:, 0, 3]
        if mode == "Randomized":
            out += blocked.sum(axis=-1)[:, :, None]/4
        else:
            raise ValueError(mode)
    return out


def simulate(sigma):
    coin = make_coin(Q_P, Q_C)
    np.testing.assert_allclose(coin.sum(axis=0), 1, rtol=0, atol=1e-15)
    p0, retained_mass = initial_state(sigma)
    data = {"initial_orientation_density": p0, "times": np.array(TIMES), "coin": coin}
    audit = []
    for mode in MODES:
        p = p0.copy()
        snapshots = []
        maximum_mass_error = 0.
        minimum = float(p.min())
        late_difference = None
        previous_rho = None
        for t in range(1, TIMES[-1]+1):
            p = step(p, coin, mode)
            maximum_mass_error = max(maximum_mass_error, abs(float(p.sum())-1))
            minimum = min(minimum, float(p.min()))
            if t == TIMES[-1]-1:
                previous_rho = p.sum(axis=2)
            if t in TIMES:
                snapshots.append(p.sum(axis=2))
        if not np.all(np.isfinite(p)) or minimum < -1e-14 or maximum_mass_error > 1e-9:
            raise FloatingPointError(f"{mode}: probability validation failed")
        late_difference = float(abs(snapshots[-1]-previous_rho).sum())
        data[mode.lower()] = np.array(snapshots)
        audit.append({"label": mode, "minimum_probability": minimum,
                      "maximum_mass_error": maximum_mass_error,
                      "final_one_step_density_L1_change": late_difference,
                      "density_excess_min_at_final_time": float((snapshots[-1]*N*N-1).min()),
                      "density_excess_max_at_final_time": float((snapshots[-1]*N*N-1).max())})
        print(f"{mode}: complete to t={TIMES[-1]}, mass error={maximum_mass_error:.3g}, "
              f"minimum={minimum:.3g}, final density step change={late_difference:.3g}", flush=True)
    data["simulation_metadata_json"] = np.array(json.dumps({
        "figure": 6, "q_p": Q_P, "q_c": Q_C, "N": N, "times": list(TIMES),
        "ell": 1., "delta_t": 1., "initial_source": {
            "type": "Gaussian cell integrals conditioned on central 5x5 sites",
            "center_site": list(ORIGIN), "support_sites_per_axis": [18, 19, 20, 21, 22],
            "sigma_before_truncation": sigma, "unconditioned_mass_in_patch": retained_mass,
            "orientation_vector": [.25]*4},
        "boundary_rules": {
            "Specular": "Outward blocked cardinal heading reflects across that wall's normal; probability stays at its departure site.",
            "Randomized": "Outward blocked probability stays at its departure site and splits equally among all four headings."},
        "specular_reversing_equivalence": "For cardinal headings at axis-aligned walls, the blocked heading is normal to the wall; the previous reversing wall coin has the same update.",
        "validation": audit}))
    return data


def set_style():
    path = Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf")
    if path.exists():
        font_manager.fontManager.addfont(str(path))
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "mathtext.default": "regular",
                         "axes.labelsize": 23, "axes.titlesize": 24, "xtick.labelsize": 18,
                         "ytick.labelsize": 18, "legend.fontsize": 20,
                         "xtick.direction": "in", "ytick.direction": "in",
                         "pdf.fonttype": 42, "svg.fonttype": "none"})


def plot(data, destination):
    set_style()
    fig = plt.figure(figsize=(9.0, 5.8))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, .08, .055],
                         height_ratios=[1, .34], left=.10, right=.89,
                         bottom=.11, top=.90, hspace=.24, wspace=.12)
    cb_grid = gs[0, 3].subgridspec(2, 1, hspace=.40)
    limits = []
    for row, time in enumerate(TIMES):
        fields = [data[mode.lower()][row]*N*N-1 for mode in MODES]
        limit = max(float(np.max(abs(field))) for field in fields)
        limit = max(.1, math.ceil(limit*10)/10)
        limits.append(limit)
    split_y = ORIGIN[1]
    heat_axes = []
    for column, mode in enumerate(MODES):
        ax = fig.add_subplot(gs[0, column])
        heat_axes.append(ax)
        for row, time in enumerate(TIMES):
            excess = data[mode.lower()][row]*N*N-1
            norm = TwoSlopeNorm(vmin=-limits[row], vcenter=0., vmax=limits[row])
            image = ax.imshow(excess.T, origin="lower", cmap="RdBu_r", norm=norm,
                              interpolation="nearest", extent=(-.5, N-.5, -.5, N-.5))
            # Preserve physical coordinates/aspect; reveal only one spatial half.
            # For odd N the central lattice cell is cut into two visual halves.
            lower, upper = (split_y, N-.5) if row == 0 else (-.5, split_y)
            clip = Rectangle((-.5, lower), N, upper-lower, transform=ax.transData)
            image.set_clip_path(clip)
            if column == len(MODES)-1:
                cax = fig.add_subplot(cb_grid[row])
                colorbar = fig.colorbar(image, cax=cax, ticks=[-limits[row], limits[row]])
                colorbar.ax.tick_params(labelsize=13, pad=5, length=3)
                colorbar.outline.set_linewidth(.7)
                if row == 0:
                    colorbar.ax.set_title(r"$\rho/\bar\rho-1$", fontsize=17, pad=14)
        ax.axhline(split_y, color=".25", lw=1., ls="--")
        ax.plot(*ORIGIN, marker="x", color="black", ms=5.5, mew=1.2)
        ax.set(xlim=(-.5, N-.5), ylim=(-.5, N-.5))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(mode, fontsize=24, pad=9)
        if column == 0:
            for time, y in zip(TIMES, (.75, .25)):
                ax.text(-.10, y, rf"$t={time}$", transform=ax.transAxes,
                        rotation=90, ha="right", va="center", fontsize=21)
    ax = fig.add_subplot(gs[1, :2])
    for mode, ls, width, zorder in (("Specular", "-", 2.7, 3),
                                    ("Randomized", "--", 2.7, 4)):
        rho = data[mode.lower()][-1]
        profile = rho.mean(axis=1)*N*N-1
        ax.plot(np.arange(N), profile, color=PROFILE_COLORS[mode], ls=ls, lw=width,
                label=mode, zorder=zorder)
    ax.axhline(0, color=".6", lw=.8, zorder=0)
    ax.set(xlabel=r"$x$", ylabel=r"$E(x)$", xlim=(0, N-1))
    ax.set_ylabel(r"$E(x)$", fontsize=21, labelpad=8)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_yticks([0, .2, .4])
    ax.tick_params(axis="y", labelsize=16, pad=5)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(mode) for mode in MODES]
    ax.legend([handles[i] for i in order], list(MODES), frameon=False,
              loc="upper center", ncol=2, fontsize=17, handlelength=2., columnspacing=1.4)
    ax.text(0., 1.06, r"$E(x)=\langle\rho\rangle_y/\bar\rho-1$", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=16)
    ax.text(1., 1.06, r"$t=5000$", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=16)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(destination/f"Figure6.{extension}", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    metadata = json.loads(data["simulation_metadata_json"].item())
    metadata["boundary_rules"] = {mode: metadata["boundary_rules"][mode] for mode in MODES}
    metadata["validation"] = [entry for entry in metadata["validation"] if entry["label"] in MODES]
    metadata.update({"display": "rho / global_mean_rho - 1", "mean_rho": 1/(N*N),
                     "colour_map": "RdBu_r", "symmetric_colour_limit_per_time": dict(zip(map(str, TIMES), limits)),
                     "displayed_boundary_rules": list(MODES),
                     "layout": "two square split-time views and a shallow steady-state profile strip",
                     "heatmap_time_and_domain": {"upper_half": {"time": TIMES[0], "y_min": split_y, "y_max": N-.5},
                                                 "lower_half": {"time": TIMES[1], "y_min": -.5, "y_max": split_y}},
                     "time_split_is_not_a_physical_interface": True,
                     "spatial_aspect_preserved": True,
                     "colour_scale_shared_within_each_time_half": True,
                     "colourbar_ticks": "range endpoints only; separate bars for the two times",
                     "profile_axis_label": "E(x) = y-averaged rho / global mean rho - 1",
                     "profile": "y-averaged final density divided by global mean, minus one"})
    (destination/"figure6_steady_state_reflective_boundaries_meta.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print("Colour limits per row:", dict(zip(TIMES, limits)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sigma", type=float, default=GAUSSIAN_SIGMA)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.output_dir/"figure6_gaussian_specular_data.npz"
    if args.plot_only:
        with np.load(cache, allow_pickle=False) as stored:
            data = {key: stored[key].copy() for key in stored.files}
    else:
        data = simulate(args.sigma)
        np.savez_compressed(cache, **data)
    plot(data, args.output_dir)


if __name__ == "__main__":
    main()
