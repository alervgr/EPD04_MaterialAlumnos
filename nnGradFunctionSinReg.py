import pandas as pd
from forward import *


def nnGradFunctionSinReg(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y):
    # Paso 1: Enrollar nn_params para obtener cada uno de los theta (pesos/parámetros)
    initial_theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                                (hidden_layer_size, input_layer_size + 1), 'F')
    initial_theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                                (num_labels, hidden_layer_size + 1), 'F')

    # Paso 2: Definir variables necesarias
    m = len(y)
    y_d = pd.get_dummies(
        y.flatten())  # ¡¡IMPORTANTE!!: No aplicamos one-hot encoding (y_d = pd.get_dummies(y.flatten())) ya que solo tenemos 1 clase: spam/no spam.
    # Pero es importante transformar y a DataFrame para poder acceder fila por fila
    delta1 = np.zeros(initial_theta1.shape)  # Delta1 tendrá las mismas dimensiones que initial_theta1
    delta2 = np.zeros(initial_theta2.shape)  # Delta2 tendrá las mismas dimensiones que initial_theta2

    # Paso 3: Para cada fila
    for i in range(X.shape[0]):
        # Paso 3.1: Forward propagation
        a1, a2, a3 = forward(initial_theta1, initial_theta2, X, i)
        # Paso 3.2: Cálculo de los delta/errores (capa 1 no tiene)
        d3 = a3 - y_d.iloc[i]  # última capa
        d2 = np.multiply(np.dot(initial_theta2.T, d3), np.multiply(a2, 1 - a2))  # capa 2
        # Paso 3.3: Cálculo de las derivadas ajustando las dimensiones de los errores y las activaciones de cada capa correctamente
        delta1 = delta1 + (np.reshape(d2[1:, ], (hidden_layer_size, 1)) @ np.reshape(a1, (
            1, input_layer_size + 1)))  # IGUAL: delta1 = delta1 + d2[1:,np.newaxis] @ a1[np.newaxis, :]
        delta2 = delta2 + (np.reshape(d3.values, (num_labels, 1)) @ np.reshape(a2, (
            1, hidden_layer_size + 1)))  # IGUAL: delta2 = delta2 + d3[:,np.newaxis] @ a2[np.newaxis, :]

    # Paso 4: Se desenrollan ambas derivadas con el mismo order con el que se enrollaron
    delta1 /= m
    delta2 /= m

    #Con regulacion
    # delta1[:,1:] += (initial_theta1[:,1:] * param_lambda / m)
    # delta2[:,1:] += (initial_theta2[:,1:] * param_lambda / m)

    gradiente = np.hstack((delta1.ravel(order='F'), delta2.ravel(order='F')))
    return gradiente
