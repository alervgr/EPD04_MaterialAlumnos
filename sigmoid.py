import numpy as np


def sigmoid(z):
    g = 1 / (1 + (np.e ** (-z)))
    return g
