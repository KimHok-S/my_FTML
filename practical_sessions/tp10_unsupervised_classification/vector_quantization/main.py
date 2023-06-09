"""
Use Vector quantization for classification
"""

import numpy as np
from utils import clean_filename, create_directory_if_missing
import cv2
import matplotlib.pyplot as plt
import optuna
import os
from scipy.spatial.distance import cdist

sigma = 0.3

"""
https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
"""

X_train = np.load(os.path.join("data", "X_train.npy"), allow_pickle=True)
X_test = np.load(os.path.join("data", "X_test.npy"), allow_pickle=True)
y_train = np.load(os.path.join("data", "y_train.npy"), allow_pickle=True)
y_test = np.load(os.path.join("data", "y_test.npy"), allow_pickle=True)

N_VALIDATION = 5000
VALIDATION_SET = range(N_VALIDATION)
X_VALIDATION = X_test[VALIDATION_SET]
y_validation = y_test[VALIDATION_SET]

N_TEST = 5000
TEST_SET = range(N_VALIDATION, N_VALIDATION + N_TEST)
X_TEST = X_test[TEST_SET]
y_test = y_test[TEST_SET]

N_OPTUNA_TRIALS = 200
SIGMA_MIN = 1e-6
SIGMA_MAX = 1e0

create_directory_if_missing(os.path.join("images"))

"""
Add code here
"""
print("X_train.shape", X_train.shape)

# convert to .jpg 28x28
for i in range(0, 10):
    cv2.imwrite(os.path.join("images", "X_train_" + str(i) + ".jpg"), X_train[i].reshape(28, 28) * 255)

i = 0
for img in os.listdir("images"):
    image = cv2.imread(os.path.join("images", img))
    image = cv2.GaussianBlur(image, (5, 5), sigma)
    cv2.imwrite(os.path.join("images", str(i) + ".jpg"), image)
    i += 1

X_train = np.array([cv2.imread(os.path.join("images", str(i) + ".jpg")) for i in range(0, 10)])
# flatten images
X_train_flat = np.array([cv2.imread(os.path.join("images", str(i) + ".jpg")).flatten() for i in range(0, 10)])
print("X_train.shape", X_train.shape)

# compute distance matrix
distance_matrix = cdist(X_train_flat, X_train_flat, metric='euclidean')
print("distance_matrix.shape", distance_matrix.shape)
