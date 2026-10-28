"""Checks for the saved-data renderer; no cluster access required."""

from dataclasses import asdict
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from least_squares import scenarios, problem, simulate, initial_variance
from plot_least_squares import draw_panel, oracle_error_curves, visible_rows, make_figure
from matplotlib.legend import Legend


class PlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spectrum, x0 = problem(3)
        cls.settings = {"steps": 12, "eta": .5, "spectrum": spectrum.tolist(),
                        "x0": x0.tolist(), "scenarios": []}
        cls.series = {}
        for s in scenarios():
            rows, _, _ = simulate(s, spectrum, x0, 12, 12, .5, 42)
            cls.series[(s.panel, s.name)] = rows
            cls.settings["scenarios"].append({**asdict(s),
                "expected_E0": float(initial_variance(s, spectrum, x0).sum())})

    def tearDown(self):
        plt.close("all")

    def test_display_horizon_filters_data_not_only_axis(self):
        rows = next(iter(self.series.values()))
        self.assertEqual(len(visible_rows(rows, 4)), 5)
        with self.assertRaises(ValueError):
            visible_rows(rows, 0)
        for panel in "ABCD":
            _, ax = plt.subplots()
            draw_panel(ax, panel, self.series, self.settings, max_iter=4)
            self.assertEqual(ax.get_xlim(), (0., 4.))
            # Empty legend proxy lines are not trajectory data.
            for line in ax.lines:
                if len(line.get_xdata()):
                    self.assertLessEqual(np.max(line.get_xdata()), 4)
        # Original full trajectories remain intact.
        self.assertEqual(len(rows), 13)

    def test_false_convergence_labels_and_markers(self):
        _, ax = plt.subplots()
        draw_panel(ax, "B", self.series, self.settings)
        self.assertEqual(ax.get_ylabel(), "Expected squared norm")
        lines = {line.get_label(): line for line in ax.lines}
        self.assertEqual(lines["SGD true gradient"].get_marker(), "o")
        self.assertEqual(lines["SGD true gradient"].get_linestyle(), "None")
        self.assertEqual(lines["TSGD estimator"].get_linestyle(), "--")

    def test_each_figure_has_one_axis_one_legend_no_comments(self):
        for panel in "ABCD":
            fig = make_figure(self.series, self.settings, panel)
            self.assertEqual(len(fig.axes), 1)
            self.assertEqual(len(fig.texts), 0)
            ax = fig.axes[0]
            self.assertEqual(len(ax.texts), 0)
            self.assertEqual(len(ax.child_axes), 0)
            legends = [child for child in ax.get_children() if isinstance(child, Legend)]
            self.assertEqual(len(legends), 1)

    def test_increment_error_matches_tracking_error_accumulation(self):
        s = next(s for s in self.settings["scenarios"] if s["name"] == "TSGD B0=128" and s["panel"] == "A")
        rows = self.series[("A", s["name"])]
        k, fresh, difference = oracle_error_curves(rows, s, self.settings)
        self.assertEqual(len(k), 13)
        np.testing.assert_allclose(difference[:-1], np.diff([r["tracking_exact"] for r in rows]),
                                   atol=1e-14)
        self.assertTrue(np.all(fresh > difference))


if __name__ == "__main__":
    unittest.main()
