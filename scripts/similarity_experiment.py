"""Reproduce the controlled sinusoidal-oracle experiment in Section 3.

Run from any directory: python TSGD/scripts/similarity_experiment.py
Dependencies: numpy and matplotlib. Output paths are relative to this script.
All methods obtain g0 from two antithetic oracle calls, charged explicitly.
SGD uses that common g0 for the first step, then fresh single-sample estimates.
TSGD uses two shared-phase pointwise evaluations per subsequent update.
No transport correction is computed after the final iterate update.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def oracle(x: np.ndarray, phase: np.ndarray, omega: float) -> np.ndarray:
    return x + np.sqrt(2.0) * np.cos(omega * x + phase)


def moments(x: np.ndarray) -> tuple[float, float]:
    squared_gradient = x * x  # L = 1
    return float(np.mean(squared_gradient)), float(
        np.std(squared_gradient, ddof=1) / np.sqrt(x.size)
    )


def simulate(method: str, omega: float, rng: np.random.Generator,
             runs: int, budget: int, eta: float) -> tuple[list[dict], dict]:
    x = np.ones(runs)
    phase = rng.uniform(0.0, 2.0 * np.pi, runs)
    g = (oracle(x, phase, omega) + oracle(x, phase + np.pi, omega)) / 2.0
    initialization_error = float(np.max(np.abs(g - x)))
    if initialization_error > 2e-15:
        raise AssertionError("Antithetic initialization is not exact to roundoff")
    records = [{"method": method, "omega": omega, "iteration": 0,
                "evaluations": 0, "mean_gradient_squared": 1.0,
                "standard_error": 0.0, "analytic_mean": 1.0}]
    old_x = x.copy()
    x = x - eta * g
    cost, iteration = 2, 1
    sgd_exact = (1.0 - eta) ** 2
    max_standardized_error = 0.0
    max_gd_relative_error = 0.0
    while True:
        mean, se = moments(x)
        exact = sgd_exact if method == "SGD" else (
            (1.0 - eta) ** (2 * iteration) if omega == 0.0 else float("nan")
        )
        if method == "SGD" and se > 1e-15:
            max_standardized_error = max(max_standardized_error, abs(mean-exact)/se)
        if method == "TSGD" and omega == 0.0:
            relative = abs(mean-exact) / exact
            max_gd_relative_error = max(max_gd_relative_error, relative)
        records.append({"method": method, "omega": omega,
                        "iteration": iteration, "evaluations": cost,
                        "mean_gradient_squared": mean, "standard_error": se,
                        "analytic_mean": exact})
        next_cost = 1 if method == "SGD" else 2
        if cost + next_cost > budget:
            break
        phase = rng.uniform(0.0, 2.0 * np.pi, runs)
        if method == "SGD":
            g = oracle(x, phase, omega)
            sgd_exact = (1.0-eta)**2 * sgd_exact + eta**2
        else:
            g = g + oracle(x, phase, omega) - oracle(old_x, phase, omega)
        old_x = x
        x = x - eta * g
        cost += next_cost
        iteration += 1
    squared_gradient = x*x
    ordered = np.sort(squared_gradient)
    top_count = max(1, runs//1000)
    summary = {
        "method": method, "omega": omega, "iterations": iteration,
        "evaluations": cost, "mean_gradient_squared": mean,
        "standard_error": se,
        "squared_gradient_quantiles": {
            str(q): float(np.quantile(squared_gradient, q))
            for q in (0.5, 0.9, 0.99, 0.999)
        },
        "max_squared_gradient": float(np.max(squared_gradient)),
        "top_0.1_percent_share_of_sum": float(
            np.sum(ordered[-top_count:])/np.sum(ordered)),
        "max_initialization_error": initialization_error,
        "max_sgd_error_in_standard_errors": max_standardized_error,
        "max_exact_transport_relative_error": max_gd_relative_error,
    }
    if method == "SGD" and max_standardized_error > 6:
        raise AssertionError("SGD Monte Carlo curve is inconsistent with exact dynamics")
    if method == "TSGD" and omega == 0 and max_gd_relative_error > 1e-8:
        raise AssertionError("Exact transport disagrees with gradient descent")
    return records, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=100_000)
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()
    if args.runs < 2 or args.budget < 2:
        parser.error("runs and budget must be at least two")
    eta = 0.1
    configurations = [("SGD", 1.0)] + [("TSGD", w) for w in (0.0, 0.5, 1.0, 2.0)]
    streams = np.random.SeedSequence(args.seed).spawn(len(configurations))
    rows, summaries, curves = [], [], []
    for (method, omega), stream in zip(configurations, streams):
        records, summary = simulate(method, omega, np.random.default_rng(stream),
                                    args.runs, args.budget, eta)
        rows.extend(records)
        summaries.append(summary)
        curves.append((method, omega, records))
    out_dir = Path(__file__).resolve().parents[1] / "figures"
    out_dir.mkdir(exist_ok=True)
    with (out_dir/"similarity_experiment.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "seed": args.seed, "runs_per_curve": args.runs, "evaluation_budget": args.budget,
        "L": 1, "sigma": 1, "x0": 1, "eta": eta,
        "initialization": "Two antithetic phases phi and phi+pi; cost two for every method",
        "sampling": "Independent uniform fresh phases; one phase shared across TSGD endpoints",
        "cost": "First update costs 2; subsequent SGD/TSGD updates cost 1/2",
        "error_bands": "Monte Carlo mean plus/minus two standard errors, pointwise",
        "numpy_version": np.__version__, "matplotlib_version": matplotlib.__version__,
        "results": summaries,
    }
    (out_dir/"similarity_experiment.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    plt.rcParams.update({"font.family": "serif", "font.size": 9,
                         "axes.labelsize": 10, "legend.fontsize": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6.3, 2.45), layout="constrained")
    colors = ["#333333", "#0072B2", "#009E73", "#D55E00", "#CC79A7"]
    for color, (method, omega, records) in zip(colors, curves):
        costs = np.array([r["evaluations"] for r in records])
        means = np.array([r["mean_gradient_squared"] for r in records])
        errors = np.array([r["standard_error"] for r in records])
        label = "SGD (all $\\omega$)" if method == "SGD" else f"TSGD, $\\omega={omega:g}$"
        ax.plot(costs, means, color=color, lw=1.7,
                ls="--" if method == "SGD" else "-", label=label)
        ax.fill_between(costs, np.maximum(means-2*errors, np.finfo(float).tiny),
                        means+2*errors, color=color, alpha=0.16, linewidth=0)
    ax.set_yscale("log")
    ax.set_ylim(1e-10, 1.3)
    ax.set_xlim(0, args.budget)
    ax.set_xlabel("Stochastic-gradient oracle evaluations")
    ax.set_ylabel("Mean squared true gradient")
    ax.grid(True, which="major", alpha=0.18, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", frameon=False, ncol=2)
    fig.savefig(out_dir/"similarity_experiment.pdf", metadata={
        "Title": "Fixed noise variance, variable similarity", "CreationDate": None})
    fig.savefig(out_dir/"similarity_experiment.png", dpi=200)
    print(json.dumps({"runs": args.runs, "seed": args.seed, "results": summaries}, indent=2))


if __name__ == "__main__":
    main()
