import numpy as np
def softplus(x):
    return np.log1p(np.exp(x))

w1 = 3.4 
w2 = -5.2
w3 = -1.4
w4 = 4.2
b1 = 2.1
b2 = 3.2
b3 = -0.6

x = 0.1

h1 = x*w1 +b1
h1_out = softplus
