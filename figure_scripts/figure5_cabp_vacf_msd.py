"""Manuscript Figure 5: bulk correlations/MSD and near-wall mapping comparisons.

This is the composite formerly called Figure 18 during development. Everything
is generated from Python source; no SVG digitization, bundled numerical output
or original desktop folders are required. Full defaults reproduce the manuscript
parameters. --smoke-test runs a tiny end-to-end check, not publication results.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
from src_python_new import figure5_wall_solver as solver
from figure3_msd import simulate_batches, turn_probabilities
from figure5_plot import render

DEFAULT_OUTPUT = ROOT / "Figures -ACRW/Fig5 - Composite mapping comparison"
CASES = (("Weak turning", "weak_turning", .9, .1),
         ("Strong turning", "strong_turning", .6, .4))


def generate_bulk(output, particles, steps, seed, recompute=False):
    cache = output / "figure5_bulk_data.npz"
    batches = 10 if particles % 10 == 0 and particles >= 1000 else 2
    if particles < batches or particles % batches:
        raise ValueError("Bulk particle count must be positive and divisible by two.")
    config = {"particles": particles, "steps": steps, "seed": seed, "batches": batches,
              "cases": [list(case) for case in CASES], "ell": 1., "delta_t": 1.}
    signature = json.dumps(config, sort_keys=True)
    if cache.exists() and not recompute:
        with np.load(cache, allow_pickle=False) as saved:
            if str(saved["configuration_json"]) == signature:
                return {key: saved[key].copy() for key in saved.files}, config
    data = {}
    times = np.unique(np.r_[np.arange(1, min(20, steps)+1),
                            np.geomspace(min(20, steps), steps, 35).astype(int)])
    lag_max = min(65, steps)
    for i, (label, slug, qp, qc) in enumerate(CASES):
        msd, diag, _ = simulate_batches(turn_probabilities(qp, qc), batches,
                                        particles//batches, steps, lag_max, seed+i*100_000)
        data[f"bulk_{slug}_msd_t"] = times
        data[f"bulk_{slug}_msd"] = msd.mean(axis=0)[times]
        data[f"bulk_{slug}_vacf_t"] = np.arange(1, lag_max+1)
        data[f"bulk_{slug}_vacf"] = diag.mean(axis=0)
        print(f"Bulk {label}: {particles} trajectories, {steps} updates", flush=True)
    data["configuration_json"] = np.array(signature)
    np.savez_compressed(cache, **data)
    return data, config


def continuum_case(config):
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    cache = output / f"figure5_cabp_continuum_{config['slug']}_specular.npz"
    signature = json.dumps({k: v for k, v in config.items()
                            if k not in ("output_dir", "recompute")}, sort_keys=True)
    if cache.exists() and not config["recompute"]:
        with np.load(cache, allow_pickle=False) as saved:
            if str(saved["configuration_json"]) == signature:
                return {key: saved[key].copy() for key in saved.files}
    result = solver.continuum(config)
    snapshots = np.array([result["snapshots"][t] for t in config["times"]])
    source = solver.gaussian_spatial_mass(config, 1)
    if (not np.all(np.isfinite(snapshots)) or snapshots.min() < -1e-14
            or abs(result["survival"]-1).max() > 1e-8
            or abs(snapshots[0]-source).sum() > 1e-12):
        raise FloatingPointError("Continuum-map probability validation failed")
    data = {"times": np.array(config["times"]), "cabp_snapshots": snapshots,
            "cabp_survival": result["survival"],
            "configuration_json": np.array(signature),
            "solver_json": np.array(json.dumps(result["solver"]))}
    np.savez_compressed(cache, **data)
    return data


def generate_all(args, output):
    particles = 100 if args.smoke_test else args.particles
    steps = 30 if args.smoke_test else 1000
    data, bulk_config = generate_bulk(output, particles, steps, args.seed, args.recompute)
    data.pop("configuration_json", None)
    source = {"bulk_generation": "fresh seeded Monte Carlo, not digitized SVG markers",
              "smoke_test": args.smoke_test,
              "figure5_metadata": {**bulk_config, "N": particles, "T": steps+1,
                  "walkers": [{"label": label, "q_p": qp, "q_c": qc}
                              for label, _, qp, qc in CASES]}}
    wall_configs, leading_configs, audit = [], [], []
    for label, slug, qp, qc in CASES:
        lam = complex(qp, qc)
        config = {"label": label, "slug": slug,
                  "dr": -math.log(abs(lam)), "omega": math.atan2(qc, qp),
                  "tau": 1., "box": 15 if args.smoke_test else 151,
                  "origin": [7, 3] if args.smoke_test else [75, 10],
                  "source_sigma": (1. if args.smoke_test else 5.)/math.sqrt(2*math.log(20.)),
                  "source_radius95": 1. if args.smoke_test else 5.,
                  "orientation": "uniform", "wall_rule": "specular",
                  "times": [0, 1, 2] if args.smoke_test else [0, 50, 100, 200, 500, 1000],
                  "refine": args.refine, "n_phi": args.n_phi, "dt": args.dt,
                  "workers": args.workers, "backend": args.backend,
                  "output_dir": str(output / "wall_simulations"),
                  "recompute": args.recompute, "schema": 1,
                  "physics_sha256": solver.fingerprint()}
        # The lattice coin is inferred ONLY from correlation-matched rates.
        # The leading-map cABP is compared to that SAME original lattice walk.
        wall_data, _, checks = solver.run_case(config)
        wall_configs.append(config)
        data[f"wall_{slug}_times"] = wall_data["times"]
        data[f"wall_{slug}_lattice"] = wall_data["lattice_snapshots"]
        data[f"wall_{slug}_cabp"] = wall_data["cabp_snapshots"]
        leading = {**config, "label": label+" / continuum map",
                   "dr": 1-qp, "omega": qc, "mapping": "continuum",
                   "source_q_p": qp, "source_q_c": qc}
        leading_data = continuum_case(leading)
        leading_configs.append(leading)
        data[f"wall_{slug}_cabp_continuum"] = leading_data["cabp_snapshots"]
        audit.append({"label": label, "correlation": checks,
                      "correlation_solver": json.loads(str(wall_data["solver_json"])),
                      "continuum_solver": json.loads(str(leading_data["solver_json"]))})
    source["wall_configurations"] = wall_configs
    source["wall_continuum_configurations"] = leading_configs
    source["wall_validation"] = audit
    data["source_json"] = np.array(json.dumps(source))
    return data


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--particles", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--refine", type=int, default=2, help="dx=1/refine")
    parser.add_argument("--n-phi", type=int, default=64)
    parser.add_argument("--dt", type=float, default=.2)
    parser.add_argument("--backend", choices=("numba", "numpy", "auto"), default="numba")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    if args.particles < 2 or args.particles % 2 or args.workers < 1 or args.refine < 1:
        parser.error("Need an even particle count >=2 and positive workers/refine.")
    if args.n_phi < 4 or args.n_phi % 4 or args.dt <= 0:
        parser.error("n-phi must be a positive multiple of four and dt must be positive.")
    output = args.output_dir.resolve()
    if args.smoke_test:
        output = output / "smoke_test"
    output.mkdir(parents=True, exist_ok=True)
    bundle = output / "figure5_data.npz"
    if args.plot_only:
        with np.load(bundle, allow_pickle=False) as saved:
            data = {key: saved[key].copy() for key in saved.files}
    else:
        data = generate_all(args, output)
        np.savez_compressed(bundle, **data)
    source = json.loads(str(data["source_json"]))
    render(data, source, output)
    print("Created composite manuscript Figure 5 (formerly working Figure 18).", flush=True)


if __name__ == "__main__":
    main()
