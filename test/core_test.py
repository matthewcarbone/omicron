#!/usr/bin/env python

__author__ = "Matthew Carbone"
__maintainer__ = "Matthew Carbone"
__email__ = "x94carbone@gmail.com"
__status__ = "Prototype"

import numpy as np
import matplotlib.pyplot as plt

from omicron.core.functions import gaussian, lorentzian, gaussian_tensor, \
    single_gaussian_from_params
from omicron.core.core import GaussianGenerator


VERBOSE = False


class Test_single_gaussian_from_params:

    grid_limits = [[-0.5, 0.5], [0.1, 1.0], [0.1, 1.0]]

    def __init__(self, plot=VERBOSE):
        self.g = GaussianGenerator(
            [-10, 10], 1000, 100, 10,
            TestGaussianGenerator.grid_limits, normalize=True)
        if VERBOSE:
            self.g.print_info()

        rg = self.g.get_random_gaussian()
        g = rg[0]  # random gaussian function
        rg_mean, rg_std, rg_amp = rg[1], rg[2], rg[3]
        _rg = single_gaussian_from_params(self.g.grid, rg_mean, rg_std, rg_amp)
        np.testing.assert_array_almost_equal(g, _rg)


class TestGaussianGenerator:

    grid_limits = [[-0.5, 0.5], [0.1, 1.0], [0.1, 1.0]]

    def __init__(self, plot=VERBOSE):
        self.g = GaussianGenerator(
            [-10, 10], 1000, 100, 10,
            TestGaussianGenerator.grid_limits, normalize=True)
        if VERBOSE:
            self.g.print_info()

        self.g = GaussianGenerator(
            [-10, 10], 1000, 100, 10,
            TestGaussianGenerator.grid_limits, normalize=False)
        if VERBOSE:
            self.g.print_info()

        try:
            self.g = GaussianGenerator(
                None, 1000, 100, 10,
                TestGaussianGenerator.grid_limits, grid_override=None,
                normalize=False)
        except RuntimeError:
            pass

        special_grid_limits = [[-4, 4], [0.01, 1.0], [0.1, 1.0]]

        self.g = GaussianGenerator(
            None, None, 100, 10,
            special_grid_limits,
            grid_override=np.linspace(-10, 10, 1000, endpoint=True),
            normalize=True)
        if VERBOSE:
            self.g.print_info()

        rg = self.g.get_random_gaussian()
        if plot:
            plt.plot(self.g.grid, rg[0], 'k')
            plt.savefig("test.pdf", dpi=300, bbox_inches='tight')
        dx = self.g.grid[1] - self.g.grid[0]
        np.testing.assert_almost_equal(np.sum(rg[0] * dx), 1.0)


class TestGaussianTensor:

    grid_limits = ([-0.5, 0.5], [0.1, 1.0], [0.1, 1.0])

    def __init__(self):
        self.N = 50000
        self.x = np.linspace(-10, 10, self.N)
        self.dx = self.x[1] - self.x[0]

        for __ in range(10):
            self.run()

    def run(self):
        self.axes = [5, 10]
        self.t, __ = \
            gaussian_tensor(
                self.x, self.axes, *TestGaussianTensor.grid_limits,
                normalize=True)
        np.testing.assert_equal(self.t.shape[0], self.N)
        np.testing.assert_equal(self.t.shape[1], self.axes[0])

        for ii in range(self.t.shape[1]):
            np.testing.assert_almost_equal(
                np.sum(self.t[:, ii]) * self.dx, 1.0)


class TestGaussian:

    def __init__(self):
        self.x = np.linspace(-5, 5, 10000)
        self.dx = self.x[1] - self.x[0]

        self.g1 = gaussian(self.x, 0.0, 0.1)
        self.g2 = gaussian(self.x, 3.0, 0.1)
        self.g3 = gaussian(self.x, -3.0, 0.1)
        self.g4 = gaussian(self.x, 3.0, 0.1)
        self.g4[1234] = -1

        self.run()

    def run(self):
        self.test_normalized()
        self.test_positive()

    def test_normalized(self):
        _int = np.sum(self.g1) * self.dx
        np.testing.assert_almost_equal(_int, 1.0)

        _int = np.sum(self.g2) * self.dx
        np.testing.assert_almost_equal(_int, 1.0)

        _int = np.sum(self.g3) * self.dx
        np.testing.assert_almost_equal(_int, 1.0)

    def test_positive(self):
        np.testing.assert_equal(True, np.all(self.g1 >= 0.0))
        np.testing.assert_equal(True, np.all(self.g2 >= 0.0))
        np.testing.assert_equal(True, np.all(self.g3 >= 0.0))
        np.testing.assert_equal(True, np.any(self.g4 < 0.0))


class TestLorentzian:

    def __init__(self):
        self.x = np.linspace(-2000, 2000, 5000000)
        self.dx = self.x[1] - self.x[0]

        self.l1 = lorentzian(self.x, 0.0, 0.004)
        self.l2 = lorentzian(self.x, 3.0, 0.004)
        self.l3 = lorentzian(self.x, -3.0, 0.004)
        self.l4 = lorentzian(self.x, 3.0, 0.004)
        self.l4[1234] = -1

        self.run()

    def run(self):
        self.test_normalized()
        self.test_positive()

    def test_normalized(self):
        _int = np.sum(self.l1) * self.dx
        np.testing.assert_almost_equal(_int, 1.0, decimal=5)

        _int = np.sum(self.l2) * self.dx
        np.testing.assert_almost_equal(_int, 1.0, decimal=5)

        _int = np.sum(self.l3) * self.dx
        np.testing.assert_almost_equal(_int, 1.0, decimal=5)

    def test_positive(self):
        np.testing.assert_equal(True, np.all(self.l1 >= 0.0))
        np.testing.assert_equal(True, np.all(self.l2 >= 0.0))
        np.testing.assert_equal(True, np.all(self.l3 >= 0.0))
        np.testing.assert_equal(True, np.any(self.l4 < 0.0))
