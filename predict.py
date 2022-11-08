from forward import forward
import numpy as np


def predict(theta1, theta2, X):
    arr_h = []
    for i in range(len(X)):
        a1, a2, a3 = forward(theta1, theta2, X, i)
        # La prediccion esta en las activaciones de la ultima capa (a3)
        arr_h.append(a3)

    pred1 = np.argmax(arr_h, axis=1) + 1  # Da el maximo por fila
    return pred1
