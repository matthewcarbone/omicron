#!/usr/bin/env python

__author__ = "Matthew Carbone"
__maintainer__ = "Matthew Carbone"
__email__ = "x94carbone@gmail.com"
__status__ = "Prototype"


import numpy as np

from omicron.core.functions import gaussian_tensor


class GaussianGenerator:

    def __init__(self, grid_limits, n_grid, m_samples, gaussians_per_sample,
                 limits, normalize=True):
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
        normalize : bool
            Whether or not each of the summed Gaussians is normalized. Default
            is True.
        """

        self.grid = np.linspace(grid_limits[0], grid_limits[1], n_grid,
                                endpoint=True)
        self.gtensor = gaussian_tensor(
            self.grid, [m_samples, gaussians_per_sample], *limits,
            normalize=normalize)

        # print(self.gtensor.shape)



