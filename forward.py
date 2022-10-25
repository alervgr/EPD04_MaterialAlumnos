import numpy as np

from sigmoid import sigmoid


def forward(Theta1, Theta2, X, i):
    # bias + neuronas de la capa 1
    a1 = np.hstack((np.ones(1), X[i]))
    z2 = Theta1 @ a1
    a2 = sigmoid(z2)
    # bias + neuronas de la capa 2
    a2 = np.hstack((np.ones(1), a2))
    z3 = Theta2 @ a2
    # a3 es la salida de la capa 3 (o h)
    a3 = sigmoid(z3)
    return a1, a2, a3
