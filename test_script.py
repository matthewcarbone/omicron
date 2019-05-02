#!/usr/bin/env python

__author__ = "Matthew Carbone"
__maintainer__ = "Matthew Carbone"
__email__ = "x94carbone@gmail.com"
__status__ = "Prototype"


from test import core_test

if __name__ == '__main__':
    core_test.TestGaussian()
    core_test.TestLorentzian()
    core_test.TestGaussianTensor()
    core_test.TestGaussianGenerator()
    core_test.Test_single_gaussian_from_params()
