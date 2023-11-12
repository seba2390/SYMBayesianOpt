from typing import *

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

from src.Hyperparameters import HyperParameters
from src.GaussianProcess import GaussianProcess


class BayesianOptimization:
    def __init__(self,
                 kernel: Callable,
                 optimization_function: Callable,
                 limits: list[tuple[float, float]]):

        self.limits = limits
        self.kernel = kernel
        self.optimization_function = optimization_function
        self.gaussian_process = GaussianProcess(kernel=self.kernel, optimization_function=self.optimization_function)
        self.best_sample, self.best_sample_value = None, None

    def acquisition_function(self, sample: HyperParameters) -> float:

        def probability_of_improvement(best_sample_value: float, mean: float, variance: float) -> float:
            C = (best_sample_value - mean) / np.sqrt(variance)
            return -norm.cdf(C)

        def expected_improvement(best_sample_value: float, mean: float, variance: float) -> float:
            C = (best_sample_value - mean) / np.sqrt(variance)
            return -(norm.pdf(C) + C * norm.cdf(C)) * np.sqrt(variance)

        def lower_confidence_bound(mean: float, variance: float, kappa: float = 0.5):
            """A small κ leads to more exploitation, whereas a large κ explores more
             high-variance points at which less is known about the value of the function."""
            return mean - kappa * np.sqrt(variance)

        mean, variance = self.gaussian_process.get_mean_and_variance(sample=sample)
        # return probability_of_improvement(best_sample_value=self.best_sample_value, mean=mean, variance=variance)
        # return expected_improvement(best_sample_value=self.best_sample_value, mean=mean, variance=variance)
        return lower_confidence_bound(mean=mean, variance=variance)

    def optimize(self, N_warmup: int, N_optimize: int):
        # Warm-up
        samples = []
        for random_sample in range(N_warmup):
            params = HyperParameters(num_params=len(self.limits), limits=self.limits)
            params.set_random_parameters()
            samples.append(params)
        self.gaussian_process.add_samples(samples=samples)
        best_idx = np.argmin(self.gaussian_process.sample_values)
        self.best_sample, self.best_sample_value = samples[best_idx], self.gaussian_process.sample_values[best_idx]

        # Optimization
        def wrapper(x: np.ndarray[float]) -> float:
            return self.acquisition_function(sample=HyperParameters(num_params=len(self.limits),
                                                                    limits=self.limits,
                                                                    parameters=x))

        for optimization_sample in range(N_optimize):
            # argmin_x
            init = HyperParameters(num_params=len(self.limits), limits=self.limits)
            init.set_random_parameters()
            res = minimize(fun=wrapper, x0=init.get_parameters().flatten(), method='COBYLA', bounds=self.limits)
            x_i = HyperParameters(num_params=len(self.limits), limits=self.limits, parameters=res.x)

            # Update samples, sample-values and covariance matrix.
            self.gaussian_process.add_samples(samples=[x_i])

            # Update best values
            best_idx = np.argmin(self.gaussian_process.sample_values)
            self.best_sample, self.best_sample_value = self.gaussian_process.covariance_matrix.samples[best_idx], \
            self.gaussian_process.sample_values[best_idx]

        return {'x': self.best_sample.get_parameters(),
                'y': self.best_sample_value}