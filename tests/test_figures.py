"""Fast regression checks; these do not replace full manuscript simulations."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "figure_scripts"))

import figure2_distribution_flux as fig2
import figure3_msd as fig3
import figure5_cabp_vacf_msd as fig5
import figure6_steady_state_reflective_boundaries as fig6
from src_python_new import figure5_wall_solver as solver


class TransportTests(unittest.TestCase):
    def test_admissibility_and_roundoff_at_diamond_edge(self):
        for qp, qc in ((.9, .1), (.6, .4), (.9, -.1), (-.5, .2)):
            p = fig3.turn_probabilities(qp, qc)
            self.assertTrue(np.all(p >= 0))
            self.assertAlmostEqual(float(p.sum()), 1.)
            self.assertAlmostEqual(p[0]-p[2], qp)
            self.assertAlmostEqual(p[1]-p[3], qc)
        for qp, qc in ((.9, .2), (np.nan, 0), (0, np.inf)):
            with self.assertRaises(ValueError):
                fig3.turn_probabilities(qp, qc)

    def test_deterministic_limits_and_both_chiral_signs(self):
        steps, lag_max = 120, 30
        times = np.arange(steps+1)
        lags = np.arange(1, lag_max+1)
        for qp, qc in ((1., 0.), (-1., 0.), (0., 1.), (0., -1.)):
            with self.subTest(qp=qp, qc=qc):
                msd, diag, off = fig3.simulate_batches(
                    fig3.turn_probabilities(qp, qc), 2, 8, steps, lag_max, 88)
                expected = (times**2 if qp == 1 else times % 2 if qp == -1
                            else np.array([0, 1, 2, 1])[times % 4])
                np.testing.assert_allclose(msd, np.broadcast_to(expected, msd.shape))
                exact = complex(qp, qc)**lags
                np.testing.assert_allclose(diag, np.broadcast_to(exact.real, diag.shape))
                np.testing.assert_allclose(off, np.broadcast_to(exact.imag, off.shape))

    def test_seed_reproducibility(self):
        args = (fig3.turn_probabilities(.5, .2), 2, 20, 120, 30, 123)
        first, second = fig3.simulate_batches(*args), fig3.simulate_batches(*args)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)

    def test_transport_factors_free_intercept_and_batch_spread(self):
        times = np.arange(121)
        msd = np.stack([4*d*times+3.2 for d in (.6, .8)])
        diag, off = np.zeros((2, 30)), np.zeros((2, 30))
        off[:, 0] = [.2, .6]
        mean, sem, even_gk = fig3.estimates(msd, diag, off)
        np.testing.assert_allclose(mean, [.7, .2], atol=1e-12)
        np.testing.assert_allclose(sem, [.1, .1], atol=1e-12)
        np.testing.assert_allclose(even_gk, .25)
        np.testing.assert_allclose(sem*math.sqrt(2), np.std([[.6, .1], [.8, .3]], axis=0, ddof=1))


class BoundaryTests(unittest.TestCase):
    def test_specular_implementations_agree_and_conserve_mass(self):
        rng = np.random.default_rng(12)
        p = rng.random((7, 6, 4))
        p /= p.sum()
        for qp, qc in ((.1, -.8), (.6, .2), (.9, .1)):
            coin = fig6.make_coin(qp, qc)
            a, b = fig2.step(p, coin), fig6.step(p, coin, "Specular")
            np.testing.assert_array_equal(a, b)
            self.assertAlmostEqual(float(a.sum()), 1.)
            self.assertGreaterEqual(float(a.min()), 0.)

    def test_specular_preserves_uniform_joint_density(self):
        p = np.full((7, 6, 4), 1/(7*6*4))
        np.testing.assert_allclose(fig2.step(p, fig2.make_coin()), p)
        np.testing.assert_allclose(fig6.step(p, fig6.make_coin(.6, .2), "Specular"), p)

    def test_corner_reflects_each_normal_and_does_not_move(self):
        p = np.zeros((5, 5, 4))
        p[0, 0, 2], p[0, 0, 3] = .3, .7
        out = fig6.step(p, np.eye(4), "Specular")
        self.assertAlmostEqual(out[0, 0, 0], .3)
        self.assertAlmostEqual(out[0, 0, 1], .7)
        self.assertAlmostEqual(float(out[0, 0].sum()), 1.)

    def test_randomization_is_over_all_four_headings(self):
        p = np.zeros((5, 5, 4))
        p[0, 2, 2] = 1.
        out = fig6.step(p, np.eye(4), "Randomized")
        np.testing.assert_array_equal(out[0, 2], [.25]*4)
        self.assertAlmostEqual(float(out.sum()), 1.)
        self.assertEqual(fig6.MODES, ("Specular", "Randomized"))
        with self.assertRaises(ValueError):
            fig6.step(p, np.eye(4), "Coin")

    def test_gaussian_source_is_central_five_by_five_and_isotropic(self):
        p, retained = fig6.initial_state()
        rho = p.sum(axis=-1)
        self.assertEqual(int(np.count_nonzero(rho)), 25)
        self.assertAlmostEqual(float(p.sum()), 1.)
        self.assertTrue(0 < retained < 1)
        np.testing.assert_array_equal(p[:, :, 0], p[:, :, 3])
        np.testing.assert_allclose(rho, rho[::-1, ::-1])


class MappingTests(unittest.TestCase):
    def test_maps_use_manuscript_parameters_and_same_original_coin(self):
        self.assertEqual([(x[2], x[3]) for x in fig5.CASES], [(.9, .1), (.6, .4)])
        for _, _, qp, qc in fig5.CASES:
            lam = complex(qp, qc)
            dr, omega = -math.log(abs(lam)), math.atan2(qc, qp)
            np.testing.assert_allclose(solver.mapped_coin(dr, omega, 1.), [qp, qc])
            self.assertGreater(qc, 0)
            # Leading-map even diffusion differs from lattice by D0=1/4.
            d_leading = (1-qp)/(2*((1-qp)**2+qc**2))
            self.assertAlmostEqual(d_leading-fig3.theory(qp, qc)[0], .25)

    def test_first_angular_harmonic_and_positive_kernel(self):
        transformed, error = solver.angular_kernel(.1, .2, .1, 128)
        weights = np.fft.irfft(transformed, n=128)
        self.assertLess(error, 1e-7)
        self.assertGreaterEqual(float(weights.min()), -1e-15)
        self.assertAlmostEqual(float(weights.sum()), 1.)
        phi = np.arange(128)*2*np.pi/128
        np.testing.assert_allclose(weights @ np.exp(1j*phi), np.exp(complex(-.1, .2)*.1), atol=1e-7)

    def test_native_gaussian_cell_integrals_match_refinement(self):
        config = {"box": 15, "origin": [7, 3], "source_sigma": .7}
        coarse = solver.gaussian_spatial_mass(config, 1)
        fine = solver.gaussian_spatial_mass(config, 2)
        np.testing.assert_allclose(fine.reshape(15, 2, 15, 2).sum(axis=(1, 3)), coarse, atol=1e-16)
        self.assertAlmostEqual(float(coarse.sum()), 1.)

    def test_observation_bins_conserve_mass_without_narrow_final_bin(self):
        rho = np.full((151, 151), 1/151**2)
        mass, density, edges = solver.observed_density(rho, 2)
        self.assertEqual(mass.shape, (76, 76))
        np.testing.assert_allclose(np.diff(edges), 151/76)
        np.testing.assert_allclose(density, 1/151**2)
        self.assertAlmostEqual(float(mass.sum()), 1.)
        level, audit = solver.hdr_level(mass, density, .9)
        self.assertGreaterEqual(audit["grid_probability_at_or_above_cutoff"], .9)
        self.assertGreater(level, 0)


if __name__ == "__main__":
    unittest.main()
