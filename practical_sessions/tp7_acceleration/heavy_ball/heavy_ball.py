"""
    heavy-ball on a strongly convex least-squares loss funtion
    The design matrix was randomly generated.
"""

import math
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    gradient,
    error,
    upper_bound_strongly_convex,
    generate_output_data,
)
import optuna


def main() -> None:
    # load the data
    data_folder = "data"
    X_path = os.path.join(data_folder, "X.npy")
    X = np.load(X_path)
    rank = np.linalg.matrix_rank(X)
    n, d = X.shape
    print(f"n: {n}")
    print(f"d: {d}")
    print(f"rank of X: {rank}")

    # generate output data
    sigma = 0
    r = np.random.default_rng()
    theta_star = r.uniform(-1, 1, size=(d, 1))
    y = generate_output_data(X, theta_star, sigma, r)

    H = (1/n) * (X.T @ X)
    val_propres, vec_propres = np.linalg.eig(H)
    L = np.max(val_propres)
    mu = np.min(val_propres)
    kappa = L / mu

    # gradient descent
    theta_0 = r.uniform(-1, 1, size=(d, 1))
    theta = theta_0
    distance = []

    t_max = 10000

    upper_bounds = []

    # step size
    gamma = 1 / L

    for t in range(t_max):
        grad = gradient(theta, H, X, y)
        theta = theta - gamma * grad
        distance.append(np.log10(np.linalg.norm(theta - theta_star)**2))
        upper_bounds.append(np.log10(upper_bound_strongly_convex(t, kappa, theta_0, theta_star)))

    # inertia
    beta = ((np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))) ** 2
    gamma = 4 / ((np.sqrt(L) + np.sqrt(mu)) ** 2)
    
    # gradient descent with inertia
    theta = theta_old = theta_0
    distance_inertia = []
    
    for t in range(t_max):
        grad = gradient(theta, H, X, y)
        temp = theta - gamma * grad + beta * (theta - theta_old)
        theta_old = theta
        theta = temp
        distance_inertia.append(np.log10(np.linalg.norm(theta - theta_star)**2))


    # plot
    plt.plot(distance, label="GD")
    plt.plot(upper_bounds, label="upper bound")
    plt.plot(distance_inertia, label="Heavy ball")
    plt.xlabel("t")
    plt.ylabel("distance to theta_star")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
