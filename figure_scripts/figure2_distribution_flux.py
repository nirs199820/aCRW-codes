"""Reproduce the supplied four-panel Figure 2 with specular wall scattering.

Free parameters, point sources, headings, normalization, palettes and arrow
sampling follow the original Figure 2. Only the boundary is replaced.
The default polarization definition reproduces the original plotted observable.
The optional outgoing flux definition uses C P, as in the manuscript.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "Figures -ACRW/Fig2 - Specular density and polarization"
N = 150
TIMES = (100, 300, 500, 1000, 2000)
ORIGINS = ((75, 75), (75, 5))
WEIGHTS = {"q_I": .1, "q_R": .8, "q_B": 0., "q_L": 0., "q_M": .1}
SHIFTS = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
CMAP = LinearSegmentedColormap.from_list("plotly_hot", [
    (0., (0., 0., 0.)), (.3, (230/255, 0., 0.)),
    (.6, (1., 210/255, 0.)), (1., (1., 1., 1.))])


def make_coin():
    eye = np.eye(4)
    return (WEIGHTS["q_I"]*eye + WEIGHTS["q_R"]*np.roll(eye, -1, axis=0)
            + WEIGHTS["q_B"]*np.roll(eye, 2, axis=0)
            + WEIGHTS["q_L"]*np.roll(eye, 1, axis=0)
            + WEIGHTS["q_M"]*np.ones((4, 4))/4)


def step(p, coin):
    q = np.einsum("ij,xyj->xyi", coin, p)
    out = np.zeros_like(q)
    out[1:, :, 0] = q[:-1, :, 0]
    out[:, 1:, 1] = q[:, :-1, 1]
    out[:-1, :, 2] = q[1:, :, 2]
    out[:, :-1, 3] = q[:, 1:, 3]
    # Only rejected normal headings reflect. Position stays at that wall cell.
    out[-1, :, 2] += q[-1, :, 0]
    out[0, :, 0] += q[0, :, 2]
    out[:, -1, 3] += q[:, -1, 1]
    out[:, 0, 1] += q[:, 0, 3]
    return out


def simulate():
    coin = make_coin()
    np.testing.assert_allclose(coin.sum(axis=0), 1, rtol=0, atol=1e-15)
    np.testing.assert_allclose(coin.sum(axis=1), 1, rtol=0, atol=1e-15)
    snapshots, audit = [], []
    for name, origin in zip(("centre", "near_wall"), ORIGINS):
        p = np.zeros((N, N, 4))
        p[origin] = .25
        history = []
        mass_error = 0.
        minimum = 0.
        for t in range(1, TIMES[-1]+1):
            p = step(p, coin)
            mass_error = max(mass_error, abs(float(p.sum())-1))
            minimum = min(minimum, float(p.min()))
            if t in TIMES:
                history.append(p.copy())
                print(f"{name}: t={t}, mass error={mass_error:.3g}", flush=True)
        assert np.all(np.isfinite(p)) and minimum >= -1e-14 and mass_error < 1e-9
        snapshots.append(np.array(history))
        audit.append({"source": name, "origin": list(origin),
                      "maximum_mass_error": mass_error, "minimum_probability": minimum})
    return {"orientation_snapshots": np.array(snapshots), "coin": coin,
            "times": np.array(TIMES), "validation_json": np.array(json.dumps(audit))}


def set_style():
    font = Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "mathtext.default": "regular",
                         "axes.labelsize": 24, "xtick.labelsize": 20, "ytick.labelsize": 20,
                         "xtick.direction": "in", "ytick.direction": "in",
                         "pdf.fonttype": 42, "svg.fonttype": "none"})


def flux(p, coin, definition):
    field = np.einsum("ij,xyj->xyi", coin, p) if definition == "outgoing" else p
    jx, jy = field[:, :, 0]-field[:, :, 2], field[:, :, 1]-field[:, :, 3]
    return jx, jy, np.hypot(jx, jy)


def plot(data, time, definition, output):
    set_style()
    fig = plt.figure(figsize=(15.7, 6.2))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, .25, 1, 1],
                         height_ratios=[1, .065], wspace=.065, hspace=.18,
                         left=.025, right=.985, top=.84, bottom=.22)
    axes = [fig.add_subplot(gs[0, i]) for i in (0, 1, 3, 4)]
    index = int(np.flatnonzero(data["times"] == time)[0])
    coin = data["coin"]
    states = data["orientation_snapshots"][:, index]
    norm = LogNorm(vmin=1e-3, vmax=1)
    images = []
    for i, (ax, state, origin) in enumerate(zip(axes, [*states, *states], [*ORIGINS, *ORIGINS])):
        if i < 2:
            values = state.sum(axis=2)
            cmap = CMAP
        else:
            jx, jy, values = flux(state, coin, definition)
            cmap = "RdBu_r"
        normalized = np.clip(values/max(float(values.max()), 1e-15), 1e-3, 1)
        im = ax.imshow(normalized.T, origin="lower", cmap=cmap, norm=norm,
                       interpolation="nearest", extent=(-.5, N-.5, -.5, N-.5))
        images.append(im)
        if i >= 2:
            arrow_step = N//10
            x, y = np.meshgrid(np.arange(N), np.arange(N))
            sl = (slice(None, None, arrow_step), slice(None, None, arrow_step))
            jxq, jyq = jx.T[sl], jy.T[sl]
            magnitude = np.hypot(jxq, jyq)
            mask = magnitude > .005*max(float(magnitude.max()), 1e-15)
            ux, uy = np.zeros_like(magnitude), np.zeros_like(magnitude)
            np.divide(jxq, magnitude, out=ux, where=mask)
            np.divide(jyq, magnitude, out=uy, where=mask)
            ax.quiver(x[sl][mask], y[sl][mask], .9*arrow_step*ux[mask], .9*arrow_step*uy[mask],
                      color="white", edgecolors="black", linewidths=.6,
                      angles="xy", scale_units="xy", scale=1., width=.012,
                      headwidth=3.5, headlength=4., headaxislength=3.5, pivot="mid", zorder=5)
        ax.plot(*origin, marker="x", color="red", markersize=3.6, markeredgewidth=1.1, zorder=6)
        ax.set(xticks=[], yticks=[], xlim=(-.5, N-.5), ylim=(-.5, N-.5))
        ax.text(-.01, 1.055, f"({chr(97+i)})", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=30)
    # One colour bar per pair, matching the supplied manually assembled figure.
    vector_label = (r"$|\mathbf{J}|/|\mathbf{J}|_{\rm max}$" if definition == "outgoing"
                    else r"$|\mathbf{m}|/|\mathbf{m}|_{\rm max}$")
    for columns, image_index, label in ((slice(0, 2), 0, r"$\rho/\rho_{\rm max}$"),
                                        (slice(3, 5), 2, vector_label)):
        full = gs[1, columns].get_position(fig)
        cax = fig.add_axes([full.x0+.20*full.width, full.y0, .60*full.width, full.height])
        cb = fig.colorbar(images[image_index], cax=cax, orientation="horizontal", ticks=[1e-3, 1e-2, 1e-1, 1])
        cb.set_label(label, fontsize=25, labelpad=6)
        cb.ax.tick_params(labelsize=19, length=4, pad=9)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(output/f"Figure2.{extension}", dpi=240, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--time", type=int, choices=TIMES, default=2000)
    parser.add_argument("--flux-definition", choices=("outgoing", "polarization"), default="polarization")
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = output/"figure2_specular_data.npz"
    if cache.exists() and not args.recompute:
        with np.load(cache, allow_pickle=False) as saved:
            data = {key: saved[key] for key in saved.files}
    else:
        data = simulate()
        np.savez_compressed(cache, **data)
    plot(data, args.time, args.flux_definition, output)
    info = {"figure": 2, "boundary_rule": "specular; blocked normal heading reverses and stays at departure site",
            "headings": ["right", "up", "left", "down"], "N": N, "time": args.time,
            "ell": 1., "delta_t": 1., "weights": WEIGHTS, "q_p": .1, "q_c": -.8,
            "origins": [list(x) for x in ORIGINS], "initial_orientation": [.25]*4,
            "evolution": "deterministic lattice master equation", "flux_definition": args.flux_definition,
            "flux_formula": "J=nu ell sum_d e_d (C P)_d" if args.flux_definition == "outgoing" else "m=sum_d e_d P_d (orientation polarization)",
            "plotted_vector_symbol": "J" if args.flux_definition == "outgoing" else "m",
            "normalization": "Each density/magnitude panel divided by its own maximum; log colour scale 1e-3..1.",
            "arrow_sampling": "Every 15 sites; unit directions scaled to length 13.5; omit below 0.005 of sampled max.",
            "validation": json.loads(str(data["validation_json"])),
            "source_figure": "Manuscript Figure 2 before the specular-boundary update",
            "old_flipping_metadata": {"N": 75, "T": 5000, "boundary": "reflecting",
                                      "reflection_coin_type": "reflective_infinite_force",
                                      "equivalent_to_specular_for_this_lattice": True},
            "provenance_note": "Old flipping plot metadata have N=75, unlike supplied N=150 figure; new N=150 densities were computed."}
    (output/"figure2_metadata.json").write_text(json.dumps(info, indent=2)+"\n")
    print(f"Saved Figure2 with flux_definition={args.flux_definition}", flush=True)


if __name__ == "__main__":
    main()
