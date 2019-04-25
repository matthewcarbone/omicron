#!/usr/bin/env python

__author__ = "Matthew Carbone"
__maintainer__ = "Matthew Carbone"
__email__ = "x94carbone@gmail.com"
__status__ = "Prototype"

import numpy as np

from omicron.core.functions import gaussian, lorentzian


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
