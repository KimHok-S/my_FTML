import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

def main():
    dataset = "dataset_3"
    data_path = os.path.join("data", dataset, "data.npy")
    labels_path = os.path.join("data", dataset, "labels.npy")
    data = np.load(data_path)
    labels = np.load(labels_path)

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.33, random_state=2
    )
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    # Pipeline with linear kernel before standardization
    sgd = SGDClassifier(max_iter=1000)
    sgd.fit(X_train, y_train)
    print("linear kernel before standardization")
    print("Train score: ", sgd.score(X_train, y_train))
    print("Test score: ", sgd.score(X_test, y_test))

    # Pipeline with linear kernel after standardization
    pipeline = Pipeline([("scaler", StandardScaler()), ("sgd", SGDClassifier(max_iter=1000))])
    pipeline.fit(X_train, y_train)
    print("Pipeline with linear kernel after standardization")
    print("Train score: ", pipeline.score(X_train, y_train))
    print("Test score: ", pipeline.score(X_test, y_test))



if __name__ == "__main__":
    main()
