import numpy as np


def randInitializeWeights(hidden_layer_size, num_labels):
    # Variable a devolver tendrá dimensiones: (L_out, L_in+1) # +1 procedente de la bias
    W = np.zeros((num_labels, 1 + hidden_layer_size))

    # Se va a inicializar W de manera random para "romper" la simetría mientras se entrena la red neuronal
    epsilon_init = 0.12  # Se define un epsilon
    W = np.random.rand(num_labels, hidden_layer_size + 1) * (2 * epsilon_init) - epsilon_init
    return W
