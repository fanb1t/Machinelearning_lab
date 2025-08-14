from random import random,seed
# initialize a hidden layer in a network

def init_network(n_in, n_hid, n_out):
    nnet = list()
    #create hidden layer
    hid_layer = list()
    for h in range(n_hid):
        rand_w = [random() for i in range(n_in+1)]
        h_node = {'weights': rand_w}
        hid_layer.append(h_node)
    out_layer = list()
    
    nnet.append(hid_layer)
    nnet.append(out_layer)
    return nnet

seed(1)  # for reproducibility
nnet = init_network(2, 3, 1)
print(nnet)