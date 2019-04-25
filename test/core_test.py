#!/usr/bin/env python

__author__ = "Matthew Carbone"
__maintainer__ = "Matthew Carbone"
__email__ = "x94carbone@gmail.com"
__status__ = "Prototype"

import numpy as np

from omicron.core.functions import gaussian


class TestGaussian:

    def __init__(self):
        self.x = np.linspace(-5, 5, 10000)
        self.dx = self.x[1] - self.x[0]
        self.run()

    def run(self):
        self.test_normalized()

    def test_normalized(self):
        g = gaussian(self.x, 0.0, 0.1)
        _int = np.sum(g) * self.dx
        np.testing.assert_almost_equal(_int, 1.0)

        g = gaussian(self.x, 3.0, 0.1)
        _int = np.sum(g) * self.dx
        np.testing.assert_almost_equal(_int, 1.0)

        g = gaussian(self.x, -3.0, 0.1)
        _int = np.sum(g) * self.dx
        np.testing.assert_almost_equal(_int, 1.0)
