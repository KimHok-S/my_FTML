"""
    Assess the quality of the clustering of a given dataset using visualizations and heuristics
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
"""

import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from seaborn import pairplot
import pandas as pd
from yellowbrick.cluster import KElbowVisualizer


def main():
    # load the data
    data = np.load("data.npy")
    nbs_of_clusters = range(2, 15)
    # plot the data
    plt.scatter(data[:, 0], data[:, 1])
    plt.show()

    # plot the data with the clusters
    for nb_of_clusters in nbs_of_clusters:
        kmeans = KMeans(n_clusters=nb_of_clusters, n_init=10, max_iter=300)
        kmeans.fit(data)
        plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_)
        plt.show()

    # plot silhouette score
    silhouette_scores = []
    for nb_of_clusters in nbs_of_clusters:
        kmeans = KMeans(n_clusters=nb_of_clusters, n_init=10, max_iter=300)
        kmeans.fit(data)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))
    plt.plot(nbs_of_clusters, silhouette_scores)
    plt.show()

    # plot elbow method
    elbow_scores = []
    for nb_of_clusters in nbs_of_clusters:
        kmeans = KMeans(n_clusters=nb_of_clusters, n_init=10, max_iter=300)
        kmeans.fit(data)
        elbow_scores.append(kmeans.inertia_)
    plt.plot(nbs_of_clusters, elbow_scores)
    plt.show()

    # plot elbow method with yellowbrick
    model = KMeans(n_init=10, max_iter=300)
    kelbow_visualizer = KElbowVisualizer(model, k=nbs_of_clusters)
    kelbow_visualizer.fit(data)
    kelbow_visualizer.show()


if __name__ == "__main__":
    main()
