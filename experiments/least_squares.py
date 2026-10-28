"""Four mechanism illustrations for pure TSGD; see README.md.

Run from anywhere: python path/to/experiments/least_squares.py
No refreshes, clipping, or mixing. All plots use iterations, not oracle budgets.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import platform
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Scenario:
    panel: str
    name: str
    method: str
    kappa: float
    label_noise: float
    initialization: str
    batch: int = 1
    initial_mse: float = 0.0


def problem(dimension: int) -> tuple[np.ndarray, np.ndarray]:
    """H is diagonal; x*=0; ||H x0||^2=1 and L=1."""
    spectrum = np.linspace(0.1, 1.0, dimension)
    x0 = np.ones(dimension) / np.linalg.norm(spectrum)
    return spectrum, x0


def oracle(x, spectrum, v, epsilon, label_noise):
    """A^T(Ax-b), where A=sqrt(v) H^(1/2), b=s epsilon."""
    return v[:, None] * spectrum * x - (
        label_noise * np.sqrt(v[:, None] * spectrum) * epsilon
    )


def sample(rng_v, rng_noise, runs, dimension, kappa):
    # Separate streams and unconditional draws allow coupling across noise levels.
    v = 1.0 + kappa * (2 * rng_v.integers(0, 2, runs) - 1)
    epsilon = rng_noise.standard_normal((runs, dimension))
    return v, epsilon


def initialize(scenario, spectrum, x0, runs, rng):
    if scenario.initialization == "controlled":
        error = np.sqrt(scenario.initial_mse / spectrum.size) * rng.standard_normal(
            (runs, spectrum.size)
        )
        return spectrum * x0 + error
    if scenario.initialization != "sampled" or scenario.batch < 1:
        raise ValueError("Use sampled initialization with batch>=1, or controlled.")
    estimate = np.zeros((runs, spectrum.size))
    # Actual iid stochastic gradients, not artificial Gaussian noise with matched variance.
    for _ in range(scenario.batch):
        v = 1.0 + scenario.kappa * (2 * rng.integers(0, 2, runs) - 1)
        epsilon = rng.standard_normal(estimate.shape)
        estimate += oracle(x0, spectrum, v, epsilon, scenario.label_noise)
    return estimate / scenario.batch


def initial_variance(scenario, spectrum, x0):
    if scenario.initialization == "controlled":
        return np.full(spectrum.size, scenario.initial_mse / spectrum.size)
    return (
        scenario.kappa**2 * (spectrum * x0)**2
        + scenario.label_noise**2 * spectrum
    ) / scenario.batch


def initial_moments(scenario, spectrum, x0):
    # Per coordinate: X=E[x^2], C=E[xg], M=E[g^2]. All initial errors are centered.
    return (
        x0**2,
        spectrum * x0**2,
        (spectrum * x0)**2 + initial_variance(scenario, spectrum, x0),
    )


def advance_moments(scenario, spectrum, eta, moments):
    X, C, M = moments
    new_X = X - 2 * eta * C + eta**2 * M
    if scenario.method == "TSGD":
        r = 1 - eta * spectrum
        rho = r**2 + (eta * scenario.kappa * spectrum)**2
        return new_X, r * (C - eta * M), rho * M
    if scenario.method == "SGD":
        return (
            new_X,
            spectrum * new_X,
            (1 + scenario.kappa**2) * spectrum**2 * new_X
            + scenario.label_noise**2 * spectrum,
        )
    raise ValueError(f"Unknown method: {scenario.method}")


def analytic_metrics(moments, spectrum):
    X, C, M = moments
    return {
        "gradient": float(np.sum(spectrum**2 * X)),
        "estimator": float(np.sum(M)),
        "tracking": float(np.sum(M - 2 * spectrum * C + spectrum**2 * X)),
    }


def residual(scenario, spectrum, x0, eta):
    """Limiting true squared-gradient norm for the stable constant-step model."""
    if scenario.method == "SGD":
        return float(np.sum(
            eta * scenario.label_noise**2 * spectrum**2
            / (2 - eta * (1 + scenario.kappa**2) * spectrum)
        ))
    variance = initial_variance(scenario, spectrum, x0)
    M0 = (spectrum * x0)**2 + variance
    weights = (
        eta * scenario.kappa**2 * spectrum
        / (2 - eta * (1 + scenario.kappa**2) * spectrum)
    )
    return float(np.sum(variance + weights * M0))


def simulate(scenario, spectrum, x0, runs, steps, eta, seed):
    # Matched streams within each panel reduce irrelevant Monte Carlo differences.
    # Initialization is independent of ALL subsequent samples. Across iterations
    # samples are independent; the two endpoints of a difference share a sample.
    panel_seed = ord(scenario.panel)
    streams = np.random.SeedSequence([seed, panel_seed]).spawn(3)
    rng_init, rng_v, rng_noise = [np.random.default_rng(s) for s in streams]
    x = np.broadcast_to(x0, (runs, spectrum.size)).copy()
    g = initialize(scenario, spectrum, x0, runs, rng_init)
    moments = initial_moments(scenario, spectrum, x0)
    records = []
    for k in range(steps + 1):
        exact = analytic_metrics(moments, spectrum)
        values = {
            "gradient": np.sum((spectrum * x)**2, axis=1),
            "estimator": np.sum(g**2, axis=1),
            "tracking": np.sum((g - spectrum * x)**2, axis=1),
        }
        record = {"panel": scenario.panel, "scenario": scenario.name, "iteration": k}
        for key, value in values.items():
            record[f"{key}_mean"] = float(np.mean(value))
            record[f"{key}_se"] = float(np.std(value, ddof=1) / np.sqrt(runs))
            record[f"{key}_exact"] = exact[key]
        records.append(record)
        if k == steps:
            break
        new_x = x - eta * g
        v, epsilon = sample(rng_v, rng_noise, runs, spectrum.size, scenario.kappa)
        if scenario.method == "TSGD":
            # Evaluate BOTH endpoints using the same (v, epsilon), not independent noise.
            difference = (
                oracle(new_x, spectrum, v, epsilon, scenario.label_noise)
                - oracle(x, spectrum, v, epsilon, scenario.label_noise)
            )
            g = g + difference
        else:
            g = oracle(new_x, spectrum, v, epsilon, scenario.label_noise)
        x = new_x
        moments = advance_moments(scenario, spectrum, eta, moments)
    if not all(np.isfinite(r["gradient_mean"]) for r in records):
        raise FloatingPointError("Non-finite trajectory; inspect stepsize and parameters.")
    return records, x, g


def scenarios():
    result = [Scenario("A", "SGD", "SGD", .05, 1., "sampled")]
    result += [Scenario("A", f"TSGD B0={b}", "TSGD", .05, 1., "sampled", batch=b)
               for b in (1, 128, 512)]
    result += [
        Scenario("B", "SGD", "SGD", .9, 0., "sampled"),
        Scenario("B", "TSGD B0=128", "TSGD", .9, 0., "sampled", batch=128),
    ]
    result += [Scenario("C", f"{method} s={s:g}", method, .2, s, "controlled")
               for method in ("TSGD", "SGD") for s in (0., 1., 5.)]
    result += [Scenario("D", f"delta={delta:g}, E0={error:g}", "TSGD", delta,
                        1., "controlled", initial_mse=error)
               for delta in (.05, .9) for error in (.001, .1)]
    return result


def plot_results(results, output, steps):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    colors = ["#222222", "#d55e00", "#0072b2", "#009e73"]

    def line(ax, rows, label, color, style="-", metric="gradient", marker=None, offset=0):
        k = np.array([r["iteration"] for r in rows])
        mean = np.array([r[f"{metric}_mean"] for r in rows])
        se = np.array([r[f"{metric}_se"] for r in rows])
        # A floor is ONLY for rendering a log axis; CSV values are not clipped.
        floor = 1e-30
        ax.plot(k, np.maximum(mean, floor), label=label, color=color, ls=style,
                lw=1.7, marker=marker, markevery=(offset, max(1, steps // 10)),
                ms=5, markerfacecolor="none")
        ax.fill_between(k, np.maximum(mean - 2 * se, floor),
                        np.maximum(mean + 2 * se, floor), color=color, alpha=.10)

    for panel, ax in zip("ABCD", axes.flat):
        group = [(s, rows) for s, rows in results if s.panel == panel]
        for index, (s, rows) in enumerate(group):
            if panel == "C":
                if s.method == "TSGD":
                    line(ax, rows, f"TSGD, s={s.label_noise:g}", "#0072b2",
                         marker=["o", "s", "^"][index], offset=index*max(1, steps//30))
                else:
                    line(ax, rows, f"SGD, s={s.label_noise:g}",
                         ["#222222", "#d55e00", "#cc79a7"][index - 3], "--")
            elif panel == "D":
                line(ax, rows, rf"$\delta={s.kappa:g},\ E_0={s.initial_mse:g}$",
                     "#0072b2" if s.kappa == .05 else "#d55e00",
                     "-" if s.initial_mse == .001 else "--")
            else:
                line(ax, rows, s.name.replace("B0=", "$B_0=$"), colors[index])
                if panel == "B" and s.method == "TSGD":
                    line(ax, rows, r"TSGD $\mathbb{E}\|g_k\|^2$", colors[index], "--", "estimator")
        ax.set_yscale("log")
        ax.set_xlim(0, steps)
        ax.set_xlabel("Iteration k")
        ax.set_ylabel(r"$\mathbb{E}\|\nabla f(x_k)\|^2$")
        ax.grid(True, which="major", alpha=.18)
        ax.legend(fontsize=8, loc="best")
    axes[0, 0].set_title("A  Noisy gradients, accurate differences\nSampled initialization")
    axes[0, 1].set_title("B  Fresh gradients improve; transport can falsely converge\nSampled initialization")
    axes[1, 0].set_title("C  Label noise cancels from transport\nControlled exact initialization; TSGD curves coincide")
    axes[1, 1].set_title("D  Initialization quality and similarity\nControlled initialization")
    fig.suptitle("Pure TSGD mechanisms: common stepsize, no refreshes, no budget comparison", fontsize=12)
    fig.savefig(output / "least_squares.pdf")
    fig.savefig(output / "least_squares.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=4000)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--eta", type=float, default=.5)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    if args.runs < 2 or args.steps < 1 or args.dimension < 2:
        parser.error("Require runs>=2, steps>=1, dimension>=2.")
    # Common mean-square stability bound for all scenarios; not a universal TSGD theorem.
    if not 0 < args.eta < 2 / (1 + .9**2):
        parser.error("Require 0 < eta < 2/(1+0.9^2) for this model.")
    args.output.mkdir(parents=True, exist_ok=True)
    spectrum, x0 = problem(args.dimension)
    results, summaries, cancellation_reference = [], [], None
    max_cancellation_difference = 0.
    for s in scenarios():
        print(f"Panel {s.panel}: {s.name}", flush=True)
        rows, final_x, final_g = simulate(
            s, spectrum, x0, args.runs, args.steps, args.eta, args.seed
        )
        results.append((s, rows))
        if s.panel == "C" and s.method == "TSGD":
            if cancellation_reference is None:
                cancellation_reference = (final_x.copy(), final_g.copy(), rows)
            else:
                ref_x, ref_g, ref_rows = cancellation_reference
                difference = max(float(np.max(np.abs(final_x-ref_x))),
                                 float(np.max(np.abs(final_g-ref_g))),
                                 max(abs(a["gradient_mean"]-b["gradient_mean"])
                                     for a, b in zip(rows, ref_rows)))
                max_cancellation_difference = max(max_cancellation_difference, difference)
        final = rows[-1]
        summaries.append({**asdict(s), "expected_E0": float(np.sum(initial_variance(s, spectrum, x0))),
                          "final_gradient_mc": final["gradient_mean"],
                          "final_gradient_se": final["gradient_se"],
                          "final_gradient_exact": final["gradient_exact"],
                          "limiting_gradient_exact": residual(s, spectrum, x0, args.eta)})
    if max_cancellation_difference > 1e-10:
        raise AssertionError(f"Shared-noise cancellation check failed: {max_cancellation_difference}")
    with (args.output / "curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0][1][0]))
        writer.writeheader()
        for _, rows in results:
            writer.writerows(rows)
    metadata = {**vars(args), "output": str(args.output.resolve()),
                "numpy_version": np.__version__, "matplotlib_version": matplotlib.__version__,
                "python_version": platform.python_version(), "python_executable": sys.executable,
                "platform": platform.platform(),
                "runtime_environment": {key: os.environ.get(key) for key in (
                    "CONDA_DEFAULT_ENV", "SLURM_JOB_ID", "SLURM_CPUS_PER_TASK",
                    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
                "spectrum": spectrum.tolist(), "x0": x0.tolist(),
                "comparison": "Iterations, common constant stepsize; NOT computational efficiency",
                "uncertainty": "Pointwise mean +/- 2 Monte Carlo standard errors; not simultaneous bands",
                "max_cancellation_difference": max_cancellation_difference,
                "scenarios": summaries}
    with (args.output / "settings_and_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)
    plot_results(results, args.output, args.steps)
    print(f"Saved results to {args.output.resolve()}")
    print(f"Panel C cancellation discrepancy: {max_cancellation_difference:.3g}")


if __name__ == "__main__":
    main()
