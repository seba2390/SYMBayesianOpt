from typing import *

from src.Hyperparameters import HyperParameters
import numpy as np


class CovarianceMatrix:
    def __init__(self, kernel: Callable):
        self.kernel = kernel
        self.matrix = None
        self.samples = []

    def update_matrix(self, new_samples: list[HyperParameters]):
        # Updating samples
        self.samples.extend(new_samples)

        # Filling matrix for first time
        if self.matrix is None:
            N = len(self.samples)
            matrix = np.empty((N, N), dtype=float)
            for i in range(N):
                for j in range(i, N):
                    matrix[i, j] = self.kernel(self.samples[i], self.samples[j])
                    if i != j:
                        matrix[j, i] = matrix[i, j]
            self.matrix = matrix

        # Updating existing matrix
        else:
            N_new, N_old = len(new_samples), self.matrix.shape[0]
            N_total = N_new + N_old
            total_matrix = np.empty((N_total, N_total), dtype=float)

            # Updating total matrix w. old covariances
            total_matrix[:N_old, :N_old] = self.matrix

            # Calculating covariances between old and new samples
            for i in range(N_old):
                for j in range(N_new):
                    total_matrix[i, N_old + j] = self.kernel(self.samples[i], new_samples[j])
                    total_matrix[N_old + j, i] = total_matrix[i, N_old + j]

            # Calculating covariances between new samples
            for i in range(N_new):
                for j in range(i, N_new):
                    total_matrix[N_old + i, N_old + j] = self.kernel(new_samples[i], new_samples[j])
                    if i != j:
                        total_matrix[N_old + j, N_old + i] = total_matrix[N_old + i, N_old + j]

            self.matrix = total_matrix

    def get_matrix(self):
        if self.matrix is None:
            raise RuntimeError(
                f' "matrix" attribute has not yet been set. Method: ".update_matrix()" must be called first. ')
        return self.matrix
