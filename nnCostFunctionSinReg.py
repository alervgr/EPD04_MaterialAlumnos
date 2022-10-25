import numpy as np
import pandas as pd

from forward import *


def nnCostFunctionSinReg(nn_params_ini, input_layer_size, hidden_layer_size, num_labels, X, y):
    theta1 = np.reshape(a=nn_params_ini[:(input_layer_size + 1) * hidden_layer_size],
                        newshape=(hidden_layer_size, input_layer_size + 1), order="F")

    theta2 = np.reshape(a=nn_params_ini[(input_layer_size + 1) * hidden_layer_size:],
                        newshape=(num_labels, hidden_layer_size + 1), order="F")

    y_dummies = pd.get_dummies(y.flatten())

    costeTotal = 0

    m = len(y)

    for i in range(0, len(X)):
        a1, a2, a3 = forward(theta1, theta2, X, i)
        # Calcular coste

        costeTotal += np.sum(y_dummies.iloc[i] * np.log(a3) + (1 - y_dummies.iloc[i]) * np.log(1 - a3))

    return -(1 / m) * costeTotal
