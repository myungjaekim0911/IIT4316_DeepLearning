import torch
import numpy as np

def sigmoid(x):
    ############################################################################
    # TODO: Implement Sigmoid activation
    ############################################################################

    return 1.0 / (1.0 + np.exp(-x))

    ############################################################################
    # END TODO
    ############################################################################

    raise NotImplementedError("sigmoid not implemented")

x1 = int(input('x1: '))
x2 = int(input('x2: '))
h1 = sigmoid(7*x1-11)
h2 = sigmoid(10*x2-5)
y = sigmoid(-13*h1+12*h2-6)

print("h1: ",h1)
print("h2: ",h2)
print(y)