import numpy as np
def softplus(x):
    return np.log(1 + np.exp(x))

w1=3.4
w2=-5.2
w3=-1.4
w4=4.2
b1=2.1
b2=3.2
b3=-0.6
x=0.1

h1=x*w1+b1
h1_out=softplus(h1)

h2=x*w2+b2
h2_out=softplus(h2)

o1 = h1_out*w3+h2_out*w4+b3
o1_out = softplus(o1)
print(o1_out)