from typing import *

import numpy as np

from src.Hyperparameters import HyperParameters
from src.Covariance import CovarianceMatrix
from src.Utils import inverse

class GaussianProcess:
    def __init__(self, kernel: Callable, optimization_function: Callable):

        self.optimization_function = optimization_function
        self.kernel = kernel
        self.covariance_matrix = CovarianceMatrix(kernel=self.kernel)

        # Define sample = x, then y=f(x) is sample_value
        self.sample_values = []

    def add_samples(self, samples: list[HyperParameters]) -> None:
        self.covariance_matrix.update_matrix(new_samples=samples)
        for sample in samples:
            self.sample_values.append(self.optimization_function(sample))

    def get_mean_and_variance(self, sample: HyperParameters) -> tuple[float, float]:
        if self.covariance_matrix.matrix is None:
            raise RuntimeError(f'No samples have been given yet - expecting call to method ".add_samples()" first. ')
        kernel_vector = np.array([self.kernel(old_sample, sample) for old_sample in self.covariance_matrix.samples])
        inv_covar = inverse(matrix=self.covariance_matrix.get_matrix(), regularize=True)
        mean = np.dot(kernel_vector, np.dot(inv_covar, np.array(self.sample_values)))
        variance = self.kernel(sample, sample) - np.dot(kernel_vector, np.dot(inv_covar, np.array(kernel_vector)))
        if variance < 0:
            variance *= 0
        return mean, variance