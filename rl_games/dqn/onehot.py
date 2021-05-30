import numpy as np


def get_onehot_vector_from_index(index: int, size: int) -> np.ndarray:
    """
    >>> get_onehot_vector_from_index(0, 5)
    array([[1, 0, 0, 0, 0]])
    >>> get_onehot_vector_from_index(3, 4)
    array([[0, 0, 0, 1]])
    """
    x = np.zeros(size, dtype=int)
    x[index] = 1
    return x.reshape(1, size) # np.array([1 if i == index else 0 for i in range(size)])
