"""Run: python -m unittest discover -s TSGD/experiments -v"""

import unittest

import numpy as np

from least_squares import (
    Scenario, advance_moments, analytic_metrics, initial_moments,
    initial_variance, initialize, oracle, problem, residual, simulate,
)


class LeastSquaresTests(unittest.TestCase):
    def test_shared_sample_cancels_label_noise(self):
        rng = np.random.default_rng(10)
        spectrum, _ = problem(10)
        x, y, noise = rng.normal(size=(3, 100, 10))
        v = rng.choice([.1, 1.9], size=100)
        expected = v[:, None] * spectrum * (x-y)
        for s in (0, 1, 100):
            actual = oracle(x, spectrum, v, noise, s) - oracle(y, spectrum, v, noise, s)
            np.testing.assert_allclose(actual, expected, atol=2e-13)

    def test_exact_initialization_and_transport_is_gd(self):
        spectrum, x0 = problem(5)
        scenario = Scenario("C", "GD", "TSGD", 0., 5., "controlled")
        rows, x, g = simulate(scenario, spectrum, x0, 8, 30, .5, 42)
        expected = (1-.5*spectrum)**30 * x0
        np.testing.assert_allclose(x, np.broadcast_to(expected, x.shape), atol=2e-13)
        np.testing.assert_allclose(g, spectrum*x, atol=2e-13)
        self.assertLess(rows[-1]["tracking_mean"], 1e-24)

    def test_exact_transport_retains_initial_error(self):
        spectrum = np.ones(3)
        x0 = np.ones(3)
        scenario = Scenario("D", "persistent error", "TSGD", 0., 3., "controlled", initial_mse=.1)
        _, _, g0 = simulate(scenario, spectrum, x0, 30, 0, 1., 42)
        _, x, g = simulate(scenario, spectrum, x0, 30, 10, 1., 42)
        np.testing.assert_allclose(x, -(g0-x0), atol=1e-13)
        np.testing.assert_allclose(g, 0., atol=1e-13)

    def test_sampled_initialization_variance(self):
        spectrum, x0 = problem(4)
        scenario = Scenario("A", "batch", "TSGD", .6, 1., "sampled", batch=16)
        g = initialize(scenario, spectrum, x0, 30000, np.random.default_rng(20))
        errors = g-spectrum*x0
        expected = initial_variance(scenario, spectrum, x0)
        squares = errors**2
        tolerance = 6*np.std(squares, axis=0, ddof=1)/np.sqrt(len(g))
        self.assertTrue(np.all(np.abs(np.mean(squares, axis=0)-expected) < tolerance))
        self.assertTrue(np.all(np.abs(np.mean(errors, axis=0)) < 6*np.sqrt(expected/len(g))))

    def test_monte_carlo_matches_exact_moments(self):
        spectrum, x0 = problem(4)
        for method, init in (("SGD", "sampled"), ("TSGD", "sampled"),
                             ("SGD", "controlled"), ("TSGD", "controlled")):
            scenario = Scenario("A", method, method, .7, .5, init,
                                batch=8, initial_mse=.1)
            rows, _, _ = simulate(scenario, spectrum, x0, 15000, 40, .5, 10)
            for k in (0, 1, 10, 40):
                for metric in ("gradient", "estimator", "tracking"):
                    row = rows[k]
                    self.assertLessEqual(
                        abs(row[f"{metric}_mean"]-row[f"{metric}_exact"]),
                        6*row[f"{metric}_se"]+1e-12,
                        (method, init, k, metric),
                    )

    def test_false_convergence_and_conservation_law(self):
        spectrum, x0 = problem(5)
        eta = .5
        scenario = Scenario("B", "false convergence", "TSGD", .9, 0., "sampled", batch=128)
        moments = initial_moments(scenario, spectrum, x0)
        variance = initial_variance(scenario, spectrum, x0)
        weights = eta*scenario.kappa**2*spectrum/(2-eta*(1+scenario.kappa**2)*spectrum)
        conserved = variance+weights*moments[2]
        for _ in range(2000):
            moments = advance_moments(scenario, spectrum, eta, moments)
            X, C, M = moments
            np.testing.assert_allclose(M-2*spectrum*C+spectrum**2*X+weights*M,
                                       conserved, atol=1e-12)
        metrics = analytic_metrics(moments, spectrum)
        self.assertLess(metrics["estimator"], 1e-30)
        self.assertGreater(metrics["gradient"], .01)
        self.assertAlmostEqual(metrics["gradient"], residual(scenario, spectrum, x0, eta), places=12)

    def test_panel_c_trajectories_coincide(self):
        spectrum, x0 = problem(5)
        reference = None
        for s in (0., 1., 5.):
            scenario = Scenario("C", "cancel", "TSGD", .2, s, "controlled")
            rows, x, g = simulate(scenario, spectrum, x0, 50, 50, .5, 20)
            current = (x, g, np.array([r["gradient_mean"] for r in rows]))
            if reference is not None:
                for actual, expected in zip(current, reference):
                    np.testing.assert_allclose(actual, expected, atol=2e-13)
            reference = current


if __name__ == "__main__":
    unittest.main()
