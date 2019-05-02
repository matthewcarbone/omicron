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


def single_normalized_gaussian_from_params(x, m, s):
    """Gaussian function 1 / sqrt(2 * pi) / s * exp(-(x - m)**2 / 2 / s**2)"""

    n = len(x)
    x = x.reshape(n, 1)
    m = m.reshape(1, *m.shape)
    s = s.reshape(1, *m.shape)

    return np.sum(gaussian(x, m, s), axis=-1)


def gaussian_tensor(x, axes, m, s, a, normalize=True):
    """Generally a 3-tensor where the axes represent the following: axis 1 is
    the x-axis grid, axis 2 is the number of total functions (i.e. training
    data) and axis 3 is the number of Gaussian functions one desires to sum
    and normalize (if at all).

    Parameters
    ----------
    x : float, np.ndarray
        Value or grid to be evaluated of shape broadcastable into (N, 1, 1)
    axes : list
        Number of Gaussians in the extra dimensions (after x-axis dimension),
        must be of length 2.
    m : list
        Length 2 list: lower and upper bounds for the uniform distribution over
        the mean
    s : list
        Length 2 list: lower and upper bounds for the uniform distribution over
        the standard deviation
    a : list
        Same as m and s, but for the amplitude
    normalize : bool
        If true, will normalize final function, else the distribution will not
        be normalize. Default is True.

    Returns
    -------
    Matrix of shape len(x) x axes[0] in which the third axis was summed over
    and normalized, if desired. Also returns the tensor containing either all
    of the means, stds and amps.
    """

    np.testing.assert_equal(len(axes), 2)

    mean = np.random.rand(1, *axes) * (m[1] - m[0]) + m[0]
    np.testing.assert_equal(((m[0] <= mean) & (mean <= m[1])), True)

    std = np.random.rand(1, *axes) * (s[1] - s[0]) + s[0]
    np.testing.assert_equal(((s[0] <= std) & (std <= s[1])), True)

    amp = np.random.rand(1, *axes) * (a[1] - a[0]) + a[0]
    np.testing.assert_equal(((a[0] <= amp) & (amp <= a[1])), True)

    n = len(x)
    _mat = gaussian(x.reshape(n, 1, 1), mean, std, a=amp)

    if normalize:
        _mat = _mat / np.sqrt(2.0 * np.pi) / amp / std / axes[-1]
        amp = 1.0 / np.sqrt(2.0 * np.pi) / std / axes[-1]

    return np.sum(_mat, axis=-1), np.concatenate([mean, std, amp])


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
