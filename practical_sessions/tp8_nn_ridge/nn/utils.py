"""
    Utilities for application of SGD on the neural network.
    Fix this file.
"""

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """
    Apply ReLU non linearity
    """
    return np.maximum(x, 0)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivative of the ReLU non linearity
    """
    return np.heaviside(x, 0)


def forward_pass(X: np.ndarray, wh: np.ndarray, theta: np.ndarray) -> dict[str, np.ndarray]:
    """
    Compute the forward pass of a neural network with an output dimension
    of 1, and with only one hidden layer of m neurons.

    Return the intermediate results, that are useful for gradient computations.

    X:     (n, d) array
    (inputs: n inputs in dimension d)
    In this exercice, d=1
    Each input is thus a line vector.

    wh:    (d+1, m) array
    (weights between the input layer and the hidden layer)

    theta: (m+1, 1) array
    (weights between the hidden layer and the output)
    """

    if X.shape:
        n = X.shape[0]
    else:
        n = 1


    """
    FIX THIS FUNCTION
    """
    # stack X with a column of 1s in order to add the intercepts
    ones_X = np.ones(shape=(n, 1))
    X_stacked = np.column_stack((X, ones_X))

    # linear product between inputs and first hidden layer
    pre_h = np.dot(X_stacked, wh)

    # apply non linearity
    h = relu(pre_h)

    # stack h with a column of 1s in order to add the intercepts
    ones_h = np.ones(shape=(n, 1))
    h_stacked = np.column_stack((h, ones_h))

    # linear operation between hidden layer and output layer
    pre_y = np.dot(h_stacked, theta)

    # apply non linearity
    y_hat = relu(pre_y)

    # return all the steps (useful for gradients)
    outputs = dict()
    print("pre_h", pre_h.shape)
    print("h", h.shape)
    print("pre_y", pre_y.shape)
    print("y_hat", y_hat.shape)
    outputs["pre_h"] = pre_h
    outputs["h"] = h
    outputs["pre_y"] = pre_y
    outputs["y_hat"] = y_hat
    return outputs


def compute_gradients(x: np.ndarray,
              y: np.ndarray,
              pre_h: np.ndarray,
              h: np.ndarray,
              pre_y: np.ndarray,
              y_hat: np.ndarray,
              theta: np.ndarray,
              ) -> dict[str, np.ndarray]:
    """
    The gradient makes use of several intermediate
    variables returned by the forward pass, see the
    explanations in the pdf for mode details and for
    the details of the calculations.

    l is the squared los

    for instance, dl_dy_hat is the gradient
    of the loss with respect to y_hat (in this case, it is
    just a derivative).

    We use the chain rule to write the computation.

    FIX THIS FUNCTION
    """
    # first compute the gradient with respect to theta
    dl_dy_hat = y_hat - y
    dy_hat_dpre_y = relu_derivative(pre_y)
    print("dy_hat_dpre_y", dy_hat_dpre_y)
    dpre_y_dtheta = np.append(h, 1)
    dl_dtheta = dl_dy_hat * dy_hat_dpre_y * dpre_y_dtheta
    print("dl_dtheta", dl_dtheta.shape)

    # then compute the gradient with respect to wh
    dy_hat_dh = theta[:-1]
    dh_dpre_h = relu_derivative(pre_h)
    dpre_h_dwh = np.append(x, 1)
    dl_dwh = dl_dy_hat * dy_hat_dh * dh_dpre_h * dpre_h_dwh
    

    # return
    gradients = dict()
    gradients["dl_dtheta"] = dl_dtheta
    gradients["dl_dwh"] = dl_dwh
    return gradients
