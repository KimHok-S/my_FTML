import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct


def main():
    data_path = os.path.join("data", "data.npy")
    labels_path = os.path.join("data", "labels.npy")
    data = np.load(data_path)
    labels = np.load(labels_path)

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.33, random_state=2
    )
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # SVM
    clf = SVC(kernel="linear", C=1.0)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print("Linear Score: ", score)
    
    clf = SVC(kernel="sigmoid", gamma=0.1, C=1.0)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print("Sigmoïd Score: ", score)

    clf = SVC(kernel="rbf", gamma=0.1, C=1.0)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print("RBF Score: ", score)

    '''
    # Gaussian Process
    plt.figure(figsize=(10, 5))
    kernels = [1.0 * RBF(length_scale=1.15), 1.0 * DotProduct(sigma_0=1.0) ** 2]
    for i, kernel in enumerate(kernels):
        clf = GaussianProcessClassifier(kernel=kernel, random_state=0).fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print("Score: ", score)
    '''


if __name__ == "__main__":
    main()
