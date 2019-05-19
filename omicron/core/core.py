#!/usr/bin/env python

__author__ = "Matthew Carbone"
__maintainer__ = "Matthew Carbone"
__email__ = "x94carbone@gmail.com"
__status__ = "Prototype"


import numpy as np

from omicron.core.functions import gaussian_tensor
from omicron.utils.misc import current_datetime


class GaussianGenerator:

    def __init__(self, grid_limits, n_grid, m_samples, gaussians_per_sample,
                 limits, grid_override=None, normalize=True):
        """Class for generating and holding the Gaussian tensors.

        Parameters
        ----------
        grid_limits : list
            List of length 2 representing the lower and upper limits of the
            x-axis grid. The grid will be inclusive with respect to the
            endpoints.
        n_grid : int
            Number of points in the grid
        m_samples : int
            Number of samples for training on
        gaussians_per_sample : int
            The number of independently, randomly generated Gaussians to be
            added together to generate a single sample
        limits : list
            A list of length 3, each element of which is a list of length
            2, representing the:
            mean_limits : list
                List of length 2 representing the lower and upper limits of the
                means chosen for the Gaussians
            std_limits : list
                Same as above but for the standard deviations
            amp_limits : list
                Same as above but for the amplitudes
        grid_override : array_like
            Override the grid_limits with a user-inputted grid. Default is
            None (requires grid_limits).
        normalize : bool
            Whether or not each entry in the dataset is normalized to 1.
            Default is True.
        """

        if grid_override is None and grid_limits is None:
            raise RuntimeError("Either grid_override or grid_limits must be "
                               "specified.")

        if grid_override is None:
            self.grid = np.linspace(grid_limits[0], grid_limits[1], n_grid,
                                    endpoint=True)
        else:
            self.grid = grid_override

        self.gtensor, self.g_params = gaussian_tensor(
            self.grid, [m_samples, gaussians_per_sample], *limits,
            normalize=normalize)

        self.generate_stats(
            m_samples, gaussians_per_sample, limits, grid_limits, normalize)

        self.date_tag = current_datetime()

        self.get_random_gaussian()

    def get_random_gaussian(self):
        ii = np.random.randint(low=0, high=self.gtensor.shape[1] - 1)
        return self.gtensor[:, ii], self.g_params[0, ii], \
            self.g_params[1, ii], self.g_params[2, ii]

    def generate_stats(self, *args):
        """Make the statistics dictionary."""

        self.stats = {}
        m_samples, gaussians_per_sample, limits, grid_limits, normalized = args
        self.stats['m_samples'] = m_samples
        self.stats['gaussians_per_sample'] = gaussians_per_sample
        self.stats['mean_limits'] = limits[0]
        self.stats['std_limits'] = limits[1]
        self.stats['amp_limits'] = limits[2]
        self.stats['grid_limits'] = grid_limits
        self.stats['avg_means'] = np.mean(self.g_params[0])
        self.stats['avg_stds'] = np.mean(self.g_params[1])
        self.stats['avg_amps'] = np.mean(self.g_params[2])
        self.stats['normalized'] = normalized

    def print_info(self):
        print(self.date_tag)
        print("---------------------------------------")
        print("* g tensor shape             %s" % (self.gtensor.shape,))
        print("* n gaussians summed over    %i"
              % self.stats['gaussians_per_sample'])
        print("* grid of len                %i (%.02f -> %.02f)"
              % (len(self.grid), self.grid[0], self.grid[-1]))
        print("* means & limits")
        print("  - means                    %.02f & %a"
              % (self.stats['avg_means'], self.stats['mean_limits']))
        print("  - standard deviations      %.02f & %a"
              % (self.stats['avg_stds'], self.stats['std_limits']))
        print("  - amplitudes               %.02f & %a"
              % (self.stats['avg_amps'], self.stats['amp_limits']))
