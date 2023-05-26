"""
    observe the adaptivity to a low dimensional support.
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from knn_1 import predict
from constants import k
from scipy.spatial.distance import cdist


def predict(x_data: np.ndarray, y_data: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    """
        Predict with knn estimation

        Parameters:
            x_data (float matrix): (n_samples, d) samples in input space
            y_data (float vector): (n_samples, 1) values of the target function
            (here, it is the euclidean norm, for these samples)
            x_test (float matrix): (n_samples, d) data for which we
            predict a value based on the dataset.

        Returns:
            y_predictions (float matrix): predictions for the data
            in x_test.
            y_predictions must be of shape (n_samples, 1)

        You need to edit this function.
        You can use cdist from scipy.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html

    """
    # compute distances between x_test and x_data
    distances = cdist(x_test, x_data, "euclidean")

    # find the k nearest neighbors
    k_nn = np.argsort(distances, axis=1)[:, :k]

    # compute the average of the k nearest neighbors
    y_predictions = np.mean(y_data[k_nn], axis=1)

    return y_predictions


# knn without fitting
def knn(n_samples: int, x_data: np.ndarray, y_data: np.ndarray, x_data_test: np.ndarray) -> float:
    """
        Run knn with n_samples
    """
    x_data_n = x_data[:, :n_samples]
    x_data_test_n = x_data_test[:, :n_samples]

    y_data_test = np.linalg.norm(x_data_test_n, axis=1)
    # compute errors
    y_pred = predict(x_data_n, y_data, x_data_test_n)
    error = np.mean((y_pred - y_data_test)**2)

    return error


# knn using PCA
def knn_PCA(n_samples: int, x_data: np.ndarray, y_data: np.ndarray, x_data_test: np.ndarray) -> float:
    """
        Run knn with n_samples
    """
    x_data_n = x_data[:, :n_samples]
    x_data_test_n = x_data_test[:, :n_samples]

    # PCA
    pca = PCA(n_components=n_samples)
    pca.fit(x_data_n)
    x_data_n = pca.transform(x_data_n)
    x_data_test_n = pca.transform(x_data_test_n)

    y_data_test = np.linalg.norm(x_data_test_n, axis=1)
    # compute errors
    y_pred = predict(x_data_n, y_data, x_data_test_n)
    error = np.mean((y_pred - y_data_test) ** 2)

    return error


# knn using curve_fit
def knn_curvefit(n_samples: int, x_data: np.ndarray, y_data: np.ndarray, x_data_test: np.ndarray) -> float:
    """
        Run knn with n_samples
    """
    x_data_n = x_data[:, :n_samples]
    x_data_test_n = x_data_test[:, :n_samples]

    # curve_fit
    def func(x, a, b):
        return a * x + b

    popt, pcov = curve_fit(func, x_data_n, y_data)
    x_data_n = func(x_data_n, *popt)

    y_data_test = np.linalg.norm(x_data_test_n, axis=1)

    # compute errors
    y_pred = predict(x_data_n, y_data, x_data_test_n)
    error = np.mean((y_pred - y_data_test) ** 2)

    return error


def main() -> None:
    # load data
    folder = "data_knn"
    x_data = np.load(os.path.join(folder, "x_data.npy"))
    x_data_test = np.load(os.path.join(folder, "x_data_test.npy"))
    y_data = np.load(os.path.join(folder, "y_data.npy"))
    d = x_data.shape[1]

    # compute errors
    errors = []
    errors_PCA = []
    errors_curvefit = []
    for n_samples in range(1, d + 1):
        errors.append(knn(n_samples, x_data, y_data, x_data_test))
        errors_PCA.append(knn_PCA(n_samples, x_data, y_data, x_data_test))
        errors_curvefit.append(knn_curvefit(n_samples, x_data, y_data, x_data_test))

    # subplot
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].plot(range(1, d + 1), errors, label="knn")
    ax[0].set_title("knn")
    ax[0].set_xlabel("n_samples")
    ax[0].set_ylabel("error")
    ax[0].legend()

    ax[1].plot(range(1, d + 1), errors_PCA, label="knn_PCA")
    ax[1].set_title("knn_PCA")
    ax[1].set_xlabel("n_samples")
    ax[1].set_ylabel("error")
    ax[1].legend()

    ax[2].plot(range(1, d + 1), errors_curvefit, label="knn_curvefit")
    ax[2].set_title("knn_curvefit")
    ax[2].set_xlabel("n_samples")
    ax[2].set_ylabel("error")
    ax[2].legend()

    plt.show()


if __name__ == "__main__":
    main()
