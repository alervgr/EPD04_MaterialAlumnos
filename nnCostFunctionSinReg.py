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

    # Con regularizacion

    #coste_sinReg = -(1-len(X))* costeTotal
    #sum1 = np.sum(np.sum( np.power(theta1[:, 1:], 2), axis=1)) # Suma por fila
    #sum2 = np.sum(np.sum( np.power(theta2[:, 1:], 2), axis=1))
    #J = coste_sinReg + (param_lambda / (2*m)) * (sum1+sum2)

    return -(1 / m) * costeTotal
