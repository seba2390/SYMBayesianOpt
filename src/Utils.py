import numpy as np


def inverse(matrix: np.ndarray, regularize: bool) -> np.ndarray:
    if regularize:
        matrix = matrix + 1e-10 * np.eye(matrix.shape[0])

    # Numerically unstable
    # inv = np.linalg.inv(matrix)

    # More numerically stable
    U, S, V = np.linalg.svd(matrix)
    inv = np.dot(V.T, np.dot(np.diag(1.0 / S), U.T))

    return inv