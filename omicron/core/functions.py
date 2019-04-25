#!/usr/bin/env python

__author__ = "Matthew Carbone"
__maintainer__ = "Matthew Carbone"
__email__ = "x94carbone@gmail.com"
__status__ = "Prototype"


import numpy as np


def gaussian(x, m, s, a=None):
    """Gaussian function a * exp(-(x - m)**2 / 2 / s**2)

    Parameters
    ----------
    x : float, np.ndarray
        Value or grid to be evaluated
    m : float
        The mean (center) of the Gaussian
    s : float
        Standard deviation of the Gaussian
    a : float
        Prefactor, if a is None, normalizes the Gaussian, else Gaussian is not
        normalized. Default is None

    Returns
    -------
    Value or grid of the Gaussian function
    """

    if a is None:
        a = 1.0 / np.sqrt(2.0 * np.pi) / s
    return a * np.exp(-(x - m)**2 / 2.0 / s**2)


def lorentzian(x, m, s, a=None):
    """Lorentzian distribution function a * (s**2 / ((x - m)**2 + x**2))

    Parameters
    ----------
    x : float, np.ndarray
        Value or grid to be evaluated
    m : float
        The location (center) of the Lorentzian
    s : float
        Scale parameter of the Lorentzian
    a : float
        Prefactor, if a is None, normalizes the Lorentzian, else it is not
        normalized. Default is None

    Returns
    -------
    Value or grid of the Lorentzian function
    """

    if a is None:
        a = 1.0 / np.pi / s
    return a * s**2 / ((x - m)**2 + s**2)