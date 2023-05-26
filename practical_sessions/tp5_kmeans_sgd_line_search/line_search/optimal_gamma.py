import numpy as np


def compute_gamma_star(H, gradient):
    """
    Line search gamma
    Fix this function
    """
    # Solution
    gamma_star = np.linalg.norm(gradient, ord=2) ** 2 / np.matmul(np.matmul(H, gradient), gradient)
    return gamma_star
