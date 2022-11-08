import numpy as np


def sigmoid(z):  # Funcion de activacion
    g = 1 / (1 + (np.e ** (-z)))
    return g
