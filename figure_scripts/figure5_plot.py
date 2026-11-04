"""Rendering of manuscript Figure 5 (the former working Figure 18).

The companion figure5_cabp_vacf_msd.py generates data from scratch. No local
SVGs or original desktop simulation folders are needed.
"""
from pathlib import Path
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, font_manager
from matplotlib.lines import Line2D
GREEN, BLUE = "#32A769", "#3F70AD"

def cabp_msd(t, dr, omega, v=1.):
    z = complex(dr, -omega)
    return 2*v*v*np.real(t/z + np.expm1(-z*t)/(z*z))


def observed_density(rho, width=2):
    # Exact interval-overlap integration used by the Figure 17 script.
    n = rho.shape[0]
    edges = np.linspace(0., float(n), math.ceil(n/width)+1)
    cells = np.arange(n, dtype=float)
    overlap = np.maximum(0., np.minimum(edges[1:, None], cells[None, :]+1.)
                         - np.maximum(edges[:-1, None], cells[None, :]))
    mass = overlap @ rho @ overlap.T
    sizes = np.diff(edges)
    return mass, mass/(sizes[:, None]*sizes[None, :]), edges


def hdr_level(mass, density, probability=.9):
    assert np.all(np.isfinite(mass)) and mass.min() >= 0
    total = float(mass.sum())
    order = np.argsort(density.ravel())[::-1]
    cumulative = np.cumsum(mass.ravel()[order])
    j = min(int(np.searchsorted(cumulative, probability*total
                               -32*np.finfo(float).eps*total)), len(order)-1)
    level = float(density.ravel()[order[j]])
    return level, float(mass[density >= level].sum()/total), total


def wall_segments(coordinates, values, level):
    intervals = []
    start = coordinates[0] if values[0] >= level else None
    for x0, x1, y0, y1 in zip(coordinates[:-1], coordinates[1:], values[:-1], values[1:]):
        if (y0 >= level) != (y1 >= level):
            cross = x0+(level-y0)*(x1-x0)/(y1-y0)
            if y1 >= level:
                start = cross
            else:
                intervals.append((start, cross))
                start = None
    if start is not None:
        intervals.append((start, coordinates[-1]))
    return intervals


def shade(base, strength):
    return tuple(1-strength*(1-np.array(colors.to_rgb(base))))


def plot_wall(ax, data, cfg, base):
    slug = cfg["slug"]
    ox, oy = cfg["origin"]
    times = data[f"wall_{slug}_times"][1:]
    palette = [shade(base, strength) for strength in (.48, .62, .76, .89, 1.)]
    records = []
    for j, (t, color) in enumerate(zip(times, palette), 1):
        for model, ls, lw, alpha in (("lattice", "-", 3.5, .63),
                                     ("cabp", ":", 2.9, 1.),
                                     ("cabp_continuum", "--", 2.8, .95)):
            field = data[f"wall_{slug}_{model}"][j]
            mass, density, edges = observed_density(field)
            level, enclosed, total = hdr_level(mass, density)
            centers = .5*(edges[:-1]+edges[1:])
            x = np.r_[edges[0], centers, edges[-1]]-(ox+.5)
            y = np.r_[edges[0], centers, edges[-1]]
            padded = np.pad(density, 1, mode="edge")
            actual_level = level
            if level >= padded.max():
                actual_level = .5*(level+padded[padded < level].max())
            ax.contour(x, y, padded.T, levels=[actual_level], colors=[color],
                       linestyles=[ls], linewidths=[lw], alpha=alpha)
            for side, coordinate, values in (("left", y, padded[0, :]),
                      ("right", y, padded[-1, :]), ("bottom", x, padded[:, 0]),
                      ("top", x, padded[:, -1])):
                for start, end in wall_segments(coordinate, values, actual_level):
                    xx, yy = (([x[0] if side == "left" else x[-1]]*2, [start, end])
                              if side in ("left", "right") else
                              ([start, end], [y[0] if side == "bottom" else y[-1]]*2))
                    ax.plot(xx, yy, color=color, ls=ls, lw=lw, alpha=alpha, clip_on=False)
            records.append({"case": cfg["label"], "model": model, "time": int(t),
                            "mapping": {"lattice": "lattice", "cabp": "correlation", "cabp_continuum": "continuum"}[model],
                            "level": level, "enclosed_probability": enclosed,
                            "total_probability": total})
    ax.plot(0, oy+.5, marker="x", color="black", ms=9, mew=2, zorder=20)
    ax.set(xlim=(-(ox+.5), cfg["box"]-(ox+.5)), ylim=(0, cfg["box"]),
           aspect="equal", xlabel=r"$x-x_0$")
    ax.set_xticks([-60, -40, -20, 0, 20, 40, 60])
    ax.set_yticks([0, 30, 60, 90, 120, 150])
    ax.text(.5, 1.04, cfg["label"], color=base, ha="center", va="bottom",
            transform=ax.transAxes, fontsize=28)
    ax.tick_params(labelbottom=False, labelleft=False, labeltop=False, labelright=False)
    handles = [Line2D([], [], color=c, lw=3.5, label=rf"$t={t}$")
               for c, t in zip(palette, times)]
    time_legend = ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=20,
                            borderaxespad=.5, labelspacing=.28, handlelength=1.7)
    ax.add_artist(time_legend)
    return records


def render(data, source, output_dir):
    times_font = Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf")
    if times_font.exists():
        font_manager.fontManager.addfont(str(times_font))
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times",
            "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "mathtext.default": "regular",
        "font.size": 22, "axes.labelsize": 30, "axes.linewidth": 1.,
        "xtick.labelsize": 23, "ytick.labelsize": 23,
        "xtick.direction": "in", "ytick.direction": "in",
        "legend.fontsize": 21, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(16.1, 13.7))
    axes = [fig.add_axes(rect) for rect in (
        [.095, .685, .392, .275], [.593, .685, .392, .275],
        [.095, .090, .392, .392*16.1/13.7],
        [.593, .090, .392, .392*16.1/13.7])]
    am, av, aw, ast = axes
    bulk_checks = []
    for label, slug, color in (("Weak turning", "weak_turning", GREEN),
                                ("Strong turning", "strong_turning", BLUE)):
        cfg = next(c for c in source["figure5_metadata"]["walkers"] if c["label"] == label)
        qp, qc = cfg["q_p"], cfg["q_c"]
        lam = complex(qp, qc)
        rates = ((1-qp, qc, "--"), (-np.log(abs(lam)), np.angle(lam), ":"))
        for dr, omega, ls in rates:
            tm = np.geomspace(1, 1000, 1200)
            tv = np.linspace(1, 65, 1200)
            am.loglog(tm, cabp_msd(tm, dr, omega), color=color, ls=ls, lw=4., alpha=.9)
            av.plot(tv, np.exp(-dr*tv)*np.cos(omega*tv), color=color, ls=ls, lw=4., alpha=.9)
        for ax, obs in ((am, "msd"), (av, "vacf")):
            ax.scatter(data[f"bulk_{slug}_{obs}_t"], data[f"bulk_{slug}_{obs}"],
                       color=color, s=54, edgecolors="none", alpha=.9, zorder=4)
        bulk_checks.append({"label": label, "q_p": qp, "q_c": qc,
                            "continuum_mapping_Dr": rates[0][0],
                            "continuum_mapping_omega": rates[0][1],
                            "correlation_mapping_Dr": rates[1][0],
                            "correlation_mapping_omega": rates[1][1]})
    am.set(xlabel=r"$t$", ylabel=r"$\mathrm{MSD}$", xlim=(.7, 1400), ylim=(.55, 40000))
    am.set_xticks([1, 10, 100, 1000], labels=["1", "10", "100", "1000"])
    am.set_yticks([1, 10, 100, 1000, 10000], labels=["1", "10", "100", "1000", "10000"])
    am.legend(handles=[
        Line2D([], [], marker="o", color="black", ls="none", ms=9, label="aCRW"),
        Line2D([], [], color="black", ls=":", lw=3.7, label="cABP: correlation map"),
        Line2D([], [], color="black", ls="--", lw=3.7, label="cABP: continuum map")],
        loc="upper left", frameon=False, fontsize=21, handlelength=1.55,
        handletextpad=.45, labelspacing=.18, borderaxespad=.4)
    am.text(.07, .50, "Weak turning", transform=am.transAxes, color=GREEN, fontsize=27)
    am.text(.55, .32, "Strong turning", transform=am.transAxes, color=BLUE, fontsize=27)
    av.set(xlabel=r"$t$", ylabel=r"$C_V^{\mathrm{diag}}$", xlim=(1, 65), ylim=(-.4, .95))
    av.set_xticks([20, 40, 60])
    av.set_yticks([-.4, 0, .4, .8], labels=["-0.4", "0.0", "0.4", "0.8"])
    wall_checks = []
    for ax, cfg, base in zip((aw, ast), source["wall_configurations"], (GREEN, BLUE)):
        wall_checks.extend(plot_wall(ax, data, cfg, base))
    aw.set_ylabel("Position", labelpad=12)
    ast.tick_params(labelleft=False)
    wall_legend = [Line2D([], [], color=".25", lw=3.5, alpha=.63, label="aCRW"),
                   Line2D([], [], color=".25", ls=":", lw=2.9, label="cABP: correlation map"),
                   Line2D([], [], color=".25", ls="--", lw=2.8, label="cABP: continuum map")]
    ast.legend(handles=wall_legend, loc="upper left", frameon=False, fontsize=20,
              handlelength=1.8, handletextpad=.5, labelspacing=.28, borderaxespad=.5)
    for ax, letter in zip(axes, "abcd"):
        ax.tick_params(which="both", direction="in", top=True, right=True, length=5.5)
        ax.tick_params(which="minor", length=3.)
        ax.text(-.12, 1.04, f"({letter})", transform=ax.transAxes, fontsize=32)
    fig.canvas.draw()
    for extension in ("pdf", "svg", "png"):
        fig.savefig(output_dir / f"Figure5.{extension}", dpi=220)
    plt.close(fig)
    # Probability conservation in the recomputed near-wall snapshots.
    assert max(abs(r["total_probability"]-1) for r in wall_checks) < 1e-10
    metadata = {"figure": 5, "source": source, "bulk_parameters": bulk_checks,
                "wall_contours": wall_checks, "weak_color": GREEN, "strong_color": BLUE,
                "wall_mappings": ["first-harmonic correlation match", "leading continuum map"],
                "different_bulk_and_wall_parameter_sets": False,
                "all_panels_match_figure5_parameters": True}
    (output_dir / "figure5_metadata.json").write_text(json.dumps(metadata, indent=2))
    print("Created Figure5.pdf, Figure5.svg, Figure5.png and metadata.")
    for cfg in source["wall_configurations"]:
        q = np.exp(complex(-cfg["dr"], cfg["omega"])*cfg["tau"])
        print(f"Wall {cfg['label']}: tau={cfg['tau']}, qp={q.real:.9f}, qc={q.imag:.9f}")
