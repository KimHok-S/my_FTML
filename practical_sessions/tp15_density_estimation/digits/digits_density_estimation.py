"""
    Find the optimal number of components in order to 
    estimate the density of the digits dataset.

    We score each number of components with the Akaike information
    criterion.

    https://en.wikipedia.org/wiki/Akaike_information_criterion

    https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture
"""
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
import os

digits = load_digits()
X_train = digits.data
y_train = digits.target


def find_nb_components() -> int:
    """
    EDIT THIS FUNCTION

    Find the optimal number of components according
    to the Akaike information criterion (AIC).
    """
    optimal_nb_components = [i for i in range(1, 200, 5)]
    print(f"optimal_nb_components: {optimal_nb_components}")
    aics = []
    for nb_components in range(1, 200, 5):
        GMM = GaussianMixture(n_components=nb_components, covariance_type="full")
        GMM.fit(X_train)
        aics.append((GMM.aic(X_train)))
        print(f"nb_components: {nb_components}, aic: {GMM.aic(X_train)}")

    fig, ax = plt.subplots(1, 1)
    ax.plot(optimal_nb_components, aics)
    ax.set_title(f"Optimal number of components")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("AIC")
    plt.show()

    return optimal_nb_components[np.argmin(aics)]


def main() -> None:
    # search for the optimal number of components
    nb_components = find_nb_components()
    # nb_components = 121

    # fit a gaussian mixture with this number of components
    covariance_type = "full"
    GMM = GaussianMixture(n_components=nb_components, covariance_type=covariance_type)
    GMM.fit(X_train)

    # generate data according to the learned distribution
    X_generated = GMM.sample(100)
    X_generated = X_generated[0]

    # plot the generated data
    fig, ax = plt.subplots(1, 1)
    ax.imshow(X_generated.reshape(-1, 8, 8), cmap="gray")
    ax.set_title(f"Data generated from the learned distribution")
    ax.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
