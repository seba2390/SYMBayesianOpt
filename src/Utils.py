import numpy as np

from src.Hyperparameters import HyperParameters


def inverse(matrix: np.ndarray, regularize: bool) -> np.ndarray:
    if regularize:
        matrix = matrix + 1e-10 * np.eye(matrix.shape[0])

    # Numerically unstable
    # inv = np.linalg.inv(matrix)

    # More numerically stable
    U, S, V = np.linalg.svd(matrix)
    inv = np.dot(V.T, np.dot(np.diag(1.0 / S), U.T))

    return inv


def rbf_kernel(x1: HyperParameters, x2: HyperParameters, length_scale=1.0, sigma_f=1.0):
    """
    Compute the Radial Basis Function (RBF) kernel between two d-dimensional vectors.

    """
    # Compute the squared Euclidean distance between the vectors
    sqdist = np.sum((x1.get_parameters() - x2.get_parameters()) ** 2)

    # Compute the RBF kernel
    return sigma_f ** 2 * np.exp(-0.5 / length_scale ** 2 * sqdist)


# Treating variables as hyperparameters as an initial example
def rosenbrock(x: HyperParameters) -> float:
    x1, x2 = x.get_parameters(), x.get_parameters()[1]
    # return ((1 - x1) ** 2 + 100 * (x2-x1 ** 2) ** 2)[0]
    return (x1 ** 2 + x2 ** 2)[0]