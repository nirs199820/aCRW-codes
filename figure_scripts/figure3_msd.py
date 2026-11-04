"""Figure 3: trajectories/correlations plus measured transport coefficients.

Standalone free-space Monte Carlo; no walls or cABP mapping are involved.
The four parameter sets and colours match the existing Figure 3 metadata.
Positive q_c means counter-clockwise turning (right, up, left, down).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
from numba import njit, prange

WALKERS = [
    ("Random", 0.0, 0.0, "#D21517"),
    ("Chiral", 0.0, 0.5, "#3F70AD"),
    ("Active Chiral", 0.5, 0.2, "#32A769"),
    ("Anti-persistent", -0.5, 0.2, "#972B91"),
]


def theory(qp, qc):
    denominator = (1 - qp)**2 + qc**2
    return np.array([(1 - qp**2 - qc**2)/(4*denominator),
                     qc/(2*denominator)])


def turn_probabilities(qp, qc):
    """Relative heading changes 0, +pi/2, pi, -pi/2."""
    magnitude = abs(qp) + abs(qc)
    if not np.isfinite(magnitude) or magnitude > 1 + 1e-14:
        raise ValueError("The parameters must satisfy |q_p| + |q_c| <= 1.")
    probabilities = np.full(4, max(0.0, 1 - magnitude)/4)
    probabilities[0 if qp >= 0 else 2] += abs(qp)
    probabilities[1 if qc >= 0 else 3] += abs(qc)
    assert np.isclose(probabilities.sum(), 1)
    assert np.all(probabilities >= 0)
    assert np.isclose(probabilities[0] - probabilities[2], qp)
    assert np.isclose(probabilities[1] - probabilities[3], qc)
    return probabilities


@njit(cache=True, parallel=True)
def simulate_batches(probabilities, batches, particles_per_batch, steps, lag_max, seed):
    """Independent trajectory batches, retaining all time origins for correlations."""
    msd = np.zeros((batches, steps + 1))
    diag = np.zeros((batches, lag_max))
    off = np.zeros((batches, lag_max))
    cos_lookup = np.array([1, 0, -1, 0])
    sin_lookup = np.array([0, 1, 0, -1])
    cumulative = np.cumsum(probabilities)
    for batch in prange(batches):
        np.random.seed(seed + batch)
        headings = np.empty(steps + 1, dtype=np.int8)
        for particle in range(particles_per_batch):
            headings[0] = np.random.randint(0, 4)
            x, y = 0, 0
            for step in range(1, steps + 1):
                u = np.random.random()
                turn = 0
                while turn < 3 and u >= cumulative[turn]:
                    turn += 1
                heading = (headings[step - 1] + turn) % 4
                headings[step] = heading
                x += cos_lookup[heading]
                y += sin_lookup[heading]
                msd[batch, step] += x*x + y*y
            for lag in range(1, lag_max + 1):
                sdiag, soff = 0, 0
                for origin in range(steps + 1 - lag):
                    difference = (headings[origin + lag] - headings[origin]) % 4
                    sdiag += cos_lookup[difference]
                    soff += sin_lookup[difference]
                diag[batch, lag - 1] += sdiag
                off[batch, lag - 1] += soff
        msd[batch] /= particles_per_batch
        for lag in range(1, lag_max + 1):
            count = particles_per_batch*(steps + 1 - lag)
            diag[batch, lag - 1] /= count
            off[batch, lag - 1] /= count
    return msd, diag, off


def estimates(msd, diag, off, fit_min=100, cutoff=20):
    # Fit MSD = intercept + 4 D t, not an unconstrained log-log power law.
    t = np.arange(fit_min, msd.shape[1], dtype=float)
    centered = t - t.mean()
    deff = (msd[:, fit_min:] @ centered)/(4*np.dot(centered, centered))
    dodd = 0.5*off[:, :cutoff].sum(axis=1)
    dgk_even = 0.25*(1 + 2*diag[:, :cutoff].sum(axis=1))
    samples = np.column_stack([deff, dodd])
    # Independent particle batches, NOT regression errors on correlated MSD times.
    mean = samples.mean(axis=0)
    sem = samples.std(axis=0, ddof=1)/np.sqrt(samples.shape[0])
    return mean, sem, dgk_even


def style():
    font_path = Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix", "mathtext.default": "regular",
        "font.size": 23, "axes.labelsize": 30, "xtick.labelsize": 23,
        "ytick.labelsize": 23, "legend.fontsize": 22,
        "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.size": 5, "ytick.major.size": 5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })


def make_figure(data, output):
    style()
    fig = plt.figure(figsize=(12.8, 9.6))
    panels = fig.add_gridspec(2, 2, height_ratios=[1.30, 1], wspace=0.30,
                             hspace=0.46, left=0.12, right=0.98,
                             bottom=0.09, top=0.94)
    transport = panels[0, 1].subgridspec(2, 1, hspace=0.24)
    ax_msd = fig.add_subplot(panels[0, 0])
    ax_diag = fig.add_subplot(panels[1, 0])
    ax_off = fig.add_subplot(panels[1, 1], sharex=ax_diag)
    ax_even = fig.add_subplot(transport[0])
    ax_odd = fig.add_subplot(transport[1], sharex=ax_even)
    t = np.arange(data["msd"].shape[-1])
    lag_max = data["diag"].shape[-1]
    lags = np.arange(1, lag_max + 1)
    means, sems = [], []
    for i, (name, qp, qc, color) in enumerate(WALKERS):
        exact = theory(qp, qc)
        msd = data["msd"][i].mean(axis=0)
        mean, sem, _ = estimates(data["msd"][i], data["diag"][i], data["off"][i])
        means.append(mean)
        sems.append(sem)
        ax_msd.loglog(t[1:], 4*exact[0]*t[1:], "--", color=color, lw=2.3, zorder=1)
        sparse = np.unique(np.r_[np.arange(1, 16), np.geomspace(16, t[-1], 100).astype(int)])
        ax_msd.loglog(t[sparse], msd[sparse], color=color, lw=1.8, label=name, zorder=2)
        power = (qp + 1j*qc)**lags
        for ax, measured, prediction in (
            (ax_diag, data["diag"][i].mean(axis=0), power.real),
            (ax_off, data["off"][i].mean(axis=0), power.imag),
        ):
            ax.plot(lags, prediction, "--", lw=2.3, color=color, zorder=1)
            ax.plot(lags, measured, lw=1.6, color=color, zorder=2)
        for j, ax in enumerate((ax_even, ax_odd)):
            # Spread of independent batch estimates, not SEM or a confidence interval.
            sd = sem[j]*np.sqrt(data["msd"].shape[1])
            ax.scatter(i + 0.12, exact[j], marker="*", s=330, color=color,
                       edgecolors="black", linewidths=0.8, zorder=4)
            ax.errorbar(i - 0.12, mean[j], yerr=sd, fmt="o", color=color,
                        markeredgecolor="white", markeredgewidth=0.9, markersize=11,
                        elinewidth=2.0, capsize=6.0, capthick=2.0, zorder=3)
    ax_msd.set(xlabel=r"$t$", ylabel="MSD", xlim=(0.75, 650), ylim=(0.2, 1800))
    ax_msd.set_xticks([1, 10, 100], labels=["1", "10", "100"])
    ax_msd.set_yticks([1, 10, 100], labels=["1", "10", "100"])
    ax_msd.legend(frameon=False, loc="upper left", handlelength=1.2,
                  labelspacing=0.08, handletextpad=0.5, borderaxespad=0.3)
    ax_diag.set(ylabel=r"$C_V^{\rm diag}$", xlabel=r"$t$", xlim=(1, 10), ylim=(-0.56, 0.58))
    ax_off.set(ylabel=r"$C_V^{\rm off}$", xlabel=r"$t$", ylim=(-0.29, 0.59))
    for ax in (ax_diag, ax_off):
        ax.set_xticks([1, 5, 10])
        ax.set_yticks([-0.4, 0, 0.4] if ax is ax_diag else [-0.2, 0, 0.4])
    labels = ["Random", "Chiral", "Active\nChiral", "Anti-\npersistent"]
    for ax in (ax_even, ax_odd):
        ax.set_xticks(np.arange(4), labels=labels)
        ax.set_xlim(-0.5, 3.5)
        ax.tick_params(axis="x", pad=9, length=0, labelsize=20)
    ax_even.tick_params(axis="x", labelbottom=False)
    ax_even.set(ylabel=r"$D_{\rm eff}$", ylim=(-0.025, 0.75))
    ax_odd.set(ylabel=r"$D_{\rm odd}$", ylim=(-0.035, 0.415))
    ax_even.set_yticks([0, 0.2, 0.4, 0.6])
    ax_odd.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax_odd.axhline(0, color="0.7", lw=0.8, ls=":", zorder=0)
    handles = [Line2D([], [], marker="o", color="0.3", ls="none", markersize=10,
                      label=r"Measured ($\pm1$ SD)"),
               Line2D([], [], marker="*", color="0.3", ls="none", markersize=17,
                      markeredgecolor="black", markeredgewidth=0.8, label="Theory")]
    ax_even.legend(handles=handles, frameon=False, fontsize=18, loc="upper left",
                   handlelength=1.2, borderaxespad=0.5, labelspacing=0.3)
    for ax, label in [(ax_msd, "(a)"), (ax_even, "(b)"),
                      (ax_diag, "(c)"), (ax_off, "(d)")]:
        ax.text(-0.13,
                1.025 if ax is ax_msd else 1.075, label,
                transform=ax.transAxes, fontsize=31, ha="left", va="bottom")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output/f"Figure3.{suffix}", dpi=220,
                    facecolor="white")
    plt.close(fig)

    # Standalone diagnostic: convergence of the measured odd Green--Kubo sums.
    fig, ax = plt.subplots(figsize=(7.4, 4.8), layout="constrained")
    for i, (name, qp, qc, color) in enumerate(WALKERS):
        sums = 0.5*np.cumsum(data["off"][i], axis=1)
        mean = sums.mean(axis=0)
        ci = 1.96*sums.std(axis=0, ddof=1)/np.sqrt(sums.shape[0])
        ax.plot(lags, mean, color=color, lw=2, label=name)
        ax.fill_between(lags, mean-ci, mean+ci, color=color, alpha=0.15, linewidth=0)
        ax.axhline(theory(qp, qc)[1], color=color, lw=1.4, ls="--")
    ax.axvline(20, color="0.5", lw=1, ls=":")
    ax.set(xlabel=r"Correlation cutoff $K$", ylabel=r"$\widehat D_{\rm odd}^{\rm GK}(K)$",
           xlim=(1, lag_max))
    ax.legend(frameon=False, fontsize=14)
    fig.savefig(output/"odd_GK_cutoff_check.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "Figures -ACRW/Fig3 - MSD correlations and transport")
    parser.add_argument("--particles", type=int, default=100_000)
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    if (args.batches < 2 or args.particles < args.batches
            or args.particles % args.batches or args.steps < 101):
        parser.error("Need at least two equal particle batches and at least 101 steps.")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = output/"figure3_transport_data.npz"
    config = {"particles": args.particles, "batches": args.batches, "steps": args.steps,
              "seed": args.seed, "lag_max": 30, "GK_cutoff": 20, "MSD_fit_min": 100,
              "ell": 1, "delta_t": 1, "positive_qc": "counter-clockwise"}
    if cache.exists() and not args.recompute:
        with np.load(cache) as saved:
            assert json.loads(str(saved["config"])) == config, "Cache config differs; use --recompute"
            data = {key: saved[key] for key in ("msd", "diag", "off")}
    else:
        records = []
        for i, (name, qp, qc, _) in enumerate(WALKERS):
            record = simulate_batches(turn_probabilities(qp, qc), args.batches,
                                      args.particles//args.batches, args.steps,
                                      config["lag_max"], args.seed + i*100_000)
            records.append(record)
            print(f"Completed {name}: q_p={qp}, q_c={qc}", flush=True)
        data = {key: np.stack([record[i] for record in records])
                for i, key in enumerate(("msd", "diag", "off"))}
        np.savez_compressed(cache, **data, config=json.dumps(config))
    report = {"configuration": config, "walkers": [],
              "estimators": {
                  "D_eff": "Linear MSD slope/4, free intercept, t=100..steps; no exponent fit.",
                  "D_odd": "0.5 sum of measured normalized off-diagonal correlations, lags 1..20.",
                  "correlations": "All available time origins, independently generated trajectories.",
                  "uncertainty": f"Main figure error bars are +/-1 sample standard deviation across {args.batches} independent trajectory batches, each containing {args.particles//args.batches} particles; not SEM or confidence intervals. The separate cutoff diagnostic retains its explicitly documented 95% confidence band.",
                  "limitations": "Odd coefficient is a trajectory Green--Kubo estimate, not an independent wall-current response measurement."}}
    for i, (name, qp, qc, _) in enumerate(WALKERS):
        mean, sem, even_gk = estimates(data["msd"][i], data["diag"][i], data["off"][i])
        exact = theory(qp, qc)
        lam = complex(qp, qc)
        tail = (lam**21/(1-lam)).imag/2
        item = {"name": name, "q_p": qp, "q_c": qc, "theory": exact.tolist(),
                "measured": mean.tolist(), "sem": sem.tolist(),
                "sd_between_batches": (sem*np.sqrt(args.batches)).tolist(),
                "plotted_errorbar": "sd_between_batches",
                "CI95_halfwidth": (1.96*sem).tolist(),
                "even_GK_crosscheck": float(even_gk.mean()),
                "odd_theory_omitted_tail_after_20": tail,
                "measured_Cdiag_1": float(data["diag"][i, :, 0].mean()),
                "measured_Coff_1": float(data["off"][i, :, 0].mean()),
                "odd_cutoffs": {str(k): float(0.5*data["off"][i, :, :k].sum(axis=1).mean())
                                for k in (15, 20, 25, 30)}}
        assert abs(item["measured_Cdiag_1"] - qp) < max(0.002, 8/np.sqrt(args.particles*args.steps))
        assert abs(item["measured_Coff_1"] - qc) < max(0.002, 8/np.sqrt(args.particles*args.steps))
        assert abs(tail) < 2e-6
        report["walkers"].append(item)
        print(json.dumps(item), flush=True)
    (output/"figure3_transport_metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    make_figure(data, output)


if __name__ == "__main__":
    main()
