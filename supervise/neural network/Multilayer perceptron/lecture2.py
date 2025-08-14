from random import random,seed

# initialize a hidden layer in a network
def init_hidden(n_in, n_hid):
    hid_layer = list()
    for i in range(n_hid):
        rand_w = [random() for _ in range(n_in+1)]
        h_node = {'weights': rand_w}
        hid_layer.append(h_node)
    return hid_layer

seed(1)  # for reproducibility
hid_layer = init_hidden(2, 3)
print(hid_layer)