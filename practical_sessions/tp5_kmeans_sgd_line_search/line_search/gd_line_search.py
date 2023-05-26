"""
    Gradient descent (GD) on a strongly convex
    loss function.
    The design matrix is randomly generated.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from algorithms import OLS_estimator, gradient
from optimal_gamma import compute_gamma_star


def main() -> None:
    """
    Load the data
    """
    n = 60
    d = 40
    folder = "./data"
    X_path = os.path.join(folder, f"X_gaussian_n={n}_d={d}.npy")
    y_path = os.path.join(folder, f"y_n={n}_d={d}.npy")
    X = np.load(X_path)
    y = np.load(y_path)

    """
        Compute the important quantities
    """
    # Hessian matrix
    H = 1 / n * np.matmul(np.transpose(X), X)
    # compute spectrum of H
    eigenvalues, eigenvectors = np.linalg.eig(H)
    # sort the eigenvalues
    sorted_indexes = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[sorted_indexes]
    eigenvectors = eigenvectors[sorted_indexes]
    # compute strong convexity and smoothness
    L = eigenvalues[0]
    mu = eigenvalues[-1]
    kappa = L / mu
    print(f"L: {L}")
    print(f"mu: {mu}")
    # OLS estimator
    eta_star = OLS_estimator(X, y)

    """
        Preparation of the algorithms
    """
    theta_0 = np.zeros((d, 1))
    number_of_iterations = 5000
    gamma_gd = 1 / L
    GD_distances_to_opt = list()
    LS_distances_to_opt = list()

    """
        Gradient descent
    """
    theta = theta_0
    for _ in range(number_of_iterations):
        theta = theta - gamma_gd * gradient(X, y, theta)
        GD_distances_to_opt.append(np.linalg.norm(theta - eta_star, ord=2))

    """
        Line search
    """
    theta = theta_0
    for _ in range(number_of_iterations):
        gamma_star = compute_gamma_star(H, gradient(X, y, theta))
        theta = theta - gamma_star * gradient(X, y, theta)
        LS_distances_to_opt.append(np.linalg.norm(theta - eta_star, ord=2))

    """
        Plot the results
    """
    plt.figure()
    plt.plot(GD_distances_to_opt, label="GD")
    plt.plot(LS_distances_to_opt, label="LS")
    plt.legend()
    plt.xlabel("Number of iterations")
    plt.ylabel("Distance to the optimal")
    plt.title(f"GD and LS on a strongly convex loss function (kappa = {kappa})")
    plt.show()


if __name__ == "__main__":
    main()
