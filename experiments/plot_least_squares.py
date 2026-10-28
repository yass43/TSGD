"""Replot saved experiments without resimulation; also used by the notebook."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TITLES = {"A": "Initialization matters", "B": "False convergence",
          "C": "Label noise cancels", "D": "Initialization and transport errors"}
COLORS = ["#222222", "#d55e00", "#0072b2", "#009e73", "#cc79a7"]


def load_results(run_dir):
    run_dir = Path(run_dir)
    with (run_dir / "settings_and_summary.json").open(encoding="utf-8") as handle:
        settings = json.load(handle)
    series = {}
    with (run_dir / "curves.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row.pop("panel"), row.pop("scenario"))
            series.setdefault(key, []).append({
                name: int(value) if name == "iteration" else float(value)
                for name, value in row.items()
            })
    expected = {(s["panel"], s["name"]) for s in settings["scenarios"]}
    if set(series) != expected:
        raise ValueError("CSV scenarios do not match the run metadata")
    for key, rows in series.items():
        rows.sort(key=lambda row: row["iteration"])
        if [r["iteration"] for r in rows] != list(range(settings["steps"] + 1)):
            raise ValueError(f"Missing or duplicate iterations: {key}")
    return series, settings


def visible_rows(rows, max_iter):
    if max_iter is not None and max_iter < 1:
        raise ValueError("max_iter must be positive, or None for the full horizon")
    return [r for r in rows if max_iter is None or r["iteration"] <= max_iter]


def oracle_error_curves(rows, spec, settings):
    """Exact SINGLE-SAMPLE oracle errors along one TSGD trajectory distribution.

    Fresh: E||h(x_k)-grad f(x_k)||^2, not the inherited estimator error.
    Difference: E||D_{k+1}-(grad f(x_{k+1})-grad f(x_k))||^2.
    Both refer to the same x_k sequence; they are not an oracle-cost comparison.
    """
    spectrum, x0 = np.array(settings["spectrum"]), np.array(settings["x0"])
    eta, delta, s = settings["eta"], spec["kappa"], spec["label_noise"]
    if spec["initialization"] == "sampled":
        variance = (delta**2 * (spectrum*x0)**2 + s**2*spectrum) / spec["batch"]
    else:
        variance = np.full(len(spectrum), spec["initial_mse"] / len(spectrum))
    M0 = (spectrum*x0)**2 + variance
    rho = (1-eta*spectrum)**2 + (eta*delta*spectrum)**2
    k = np.array([r["iteration"] for r in rows])
    fresh = delta**2*np.array([r["gradient_exact"] for r in rows]) + s**2*spectrum.sum()
    difference = eta**2*delta**2*np.sum(spectrum**2 * M0 * rho**k[:, None], axis=1)
    return k, fresh, difference


def _positive(values):
    # Mask zero/negative values on log axes, without introducing a fake plateau.
    values = np.asarray(values)
    return np.where(values > 0, values, np.nan)


def _line(ax, rows, label, color, style="-", metric="gradient", marker=None,
          offset=0, bands=True, exact=False, zorder=2):
    k = np.array([r["iteration"] for r in rows])
    mean = np.array([r[f"{metric}_mean"] for r in rows])
    se = np.array([r[f"{metric}_se"] for r in rows])
    ax.plot(k, _positive(mean), label=label, color=color, ls=style, lw=1.7,
            marker=marker, markevery=(offset, max(1, len(rows)//9)), ms=5,
            markerfacecolor="white", markeredgewidth=1.1, zorder=zorder)
    if bands:
        ax.fill_between(k, _positive(mean-2*se), _positive(mean+2*se),
                        color=color, alpha=.10, zorder=1)
    if exact:
        ax.plot(k, _positive([r[f"{metric}_exact"] for r in rows]),
                color=color, ls=":", lw=1, alpha=.9, zorder=3)


def draw_panel(ax, panel, series, settings, max_iter=80, metric="gradient",
               exact=False, bands=True):
    if panel not in TITLES or metric not in ("gradient", "estimator", "tracking"):
        raise ValueError("Unknown panel or metric")
    specs = [s for s in settings["scenarios"] if s["panel"] == panel]
    horizon = min(settings["steps"], max_iter) if max_iter is not None else settings["steps"]
    groups = [(s, visible_rows(series[(panel, s["name"])], max_iter)) for s in specs]
    deltas = sorted({s["kappa"] for s in specs})
    errors = sorted({s["initial_mse"] for s in specs})
    for i, (s, rows) in enumerate(groups):
        color, style, marker, offset = COLORS[i % len(COLORS)], "-", None, 0
        label = s["name"]
        zorder = 2
        if panel == "A" and s["method"] == "TSGD":
            label = rf"TSGD $B_0={s['batch']}$"
        elif panel == "B":
            label = "SGD true gradient" if s["method"] == "SGD" else "TSGD true gradient"
            if metric != "gradient":
                label = s["name"]
            if s["method"] == "SGD":
                marker, zorder, style = "o", 5, "None"
        elif panel == "C":
            if s["method"] == "TSGD":
                color, marker = COLORS[2], ["o", "s", "^"][i % 3]
                offset = i*max(1, (horizon+1)//27)
                levels = ", ".join(f"{p['label_noise']:g}" for p in specs if p["method"] == "TSGD")
                label = f"TSGD (s = {levels})" if i == 0 else "_nolegend_"
            else:
                colors_sgd = [COLORS[0], COLORS[1], COLORS[4]]
                color, style = colors_sgd[(i-3) % 3], "--"
                label = f"SGD, s = {s['label_noise']:g}"
        elif panel == "D":
            color = [COLORS[2], COLORS[1]][deltas.index(s["kappa"]) % 2]
            style = ["-", "--"][errors.index(s["initial_mse"]) % 2]
            label = rf"$\delta={s['kappa']:g},\ E_0={s['initial_mse']:g}$"
        _line(ax, rows, label, color, style, metric, marker, offset, bands, exact, zorder)
        if panel == "B" and metric == "gradient" and s["method"] == "TSGD":
            _line(ax, rows, "TSGD estimator", color, "--", "estimator", bands=bands, exact=exact)
    ax.legend(fontsize=9, loc="best", framealpha=.9)
    ax.set_yscale("log")
    ax.set_xlim(0, horizon)
    ax.set_xlabel("Iteration k")
    ylabel = {"gradient": r"$\mathbb{E}\|\nabla f(x_k)\|^2$",
              "estimator": r"$\mathbb{E}\|g_k\|^2$",
              "tracking": r"$\mathbb{E}\|g_k-\nabla f(x_k)\|^2$"}[metric]
    if panel == "B" and metric == "gradient":
        ylabel = "Expected squared norm"
    ax.set_ylabel(ylabel)
    ax.set_title(f"{panel}  {TITLES[panel]}", loc="left", fontsize=12)
    ax.grid(True, which="major", alpha=.18)



def caption(settings, max_iter=80):
    horizon = settings["steps"] if max_iter is None else min(max_iter, settings["steps"])
    return (
        f"A–B: sampled initialization; C–D: controlled initialization. Common stepsize eta={settings['eta']:g}; "
        f"first {horizon} of {settings['steps']} iterations shown.\n"
        "Shading: mean +/- 2 pointwise Monte Carlo standard errors. These are mechanism illustrations, not cost comparisons."
    )


def make_figure(series, settings, panel, max_iter=80, exact=False, bands=True):
    """One standalone plot with one legend, no inset or explanatory annotations."""
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    draw_panel(ax, panel, series, settings, max_iter, exact=exact, bands=bands)
    fig.tight_layout()
    return fig


def save_figure(series, settings, output, max_iter=80, exact=False, bands=True):
    """Save four separate PDF/PNG files. Context remains in a text file, not images."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for panel in "ABCD":
        fig = make_figure(series, settings, panel, max_iter, exact, bands)
        fig.savefig(output / f"least_squares_{panel}.pdf")
        fig.savefig(output / f"least_squares_{panel}.png", dpi=200)
        plt.close(fig)
    (output / "caption.txt").write_text(caption(settings, max_iter)+"\n", encoding="utf-8")



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Default: RUN_DIR/replot (raw outputs stay untouched)")
    parser.add_argument("--max-iter", type=int, default=80, help="Displayed horizon only")
    parser.add_argument("--full-horizon", action="store_true")
    parser.add_argument("--exact", action="store_true", help="Overlay dotted exact finite-time expectations")
    parser.add_argument("--no-bands", action="store_true")
    args = parser.parse_args()
    if args.max_iter < 1:
        parser.error("--max-iter must be positive")
    plt.switch_backend("Agg")
    series, settings = load_results(args.run_dir)
    output = args.output or args.run_dir / "replot"
    save_figure(series, settings, output, None if args.full_horizon else args.max_iter,
                args.exact, not args.no_bands)
    print(f"Replotted saved data in {output.resolve()}; no simulations were run.")


if __name__ == "__main__":
    main()
