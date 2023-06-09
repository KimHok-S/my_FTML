"""
Study classifiation on data projected on a lower dimension
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import plotly.express as px

X_train = np.load(os.path.join("data", "X_train.npy"))
X_test = np.load(os.path.join("data", "X_test.npy"))
y_train = np.load(os.path.join("data", "y_train.npy"))
y_test = np.load(os.path.join("data", "y_test.npy"))

logit = LogisticRegression()
logit.fit(X_train, y_train)

print("Accuracy on train set: {:.2f}".format(logit.score(X_train, y_train)))

# plot the data in 3D
fig = px.scatter_3d(x=X_train[:, 0], y=X_train[:, 1], z=X_train[:, 2], color=y_train)
fig.show()

# PCA
for i in range(1,5):
    pca = PCA(n_components=i)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    
logit_pca = LogisticRegression()
logit_pca.fit(X_train_pca, y_train)

print("Accuracy on train set: {:.2f}".format(logit_pca.score(X_train_pca, y_train)))

# plot the data in 3D
fig = px.scatter_3d(x=X_train_pca[:, 0], y=X_train_pca[:, 1], z=X_train_pca[:, 2], color=y_train)
fig.show()

