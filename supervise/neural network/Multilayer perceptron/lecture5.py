# Set weights of your network
# do the forward pass
# and backprobpagation
from random import seed
from random import random
from math import exp

def init_network(n_in, n_hid, n_out):
    nnet = list()
    #create hidden layer
    hid_layer = list()
    for h in range(n_hid):
        rand_w = [random() for i in range(n_in+1)]
        h_node = {'weights' : rand_w}
        hid_layer.append(h_node)
    #create output layer
    out_layer = list()
    for h in range(n_out):
        rand_w = [random() for i in range(n_hid+1)]
        o_node = {'weights' : rand_w}
        out_layer.append(o_node)
    nnet.append(hid_layer)
    nnet.append(out_layer)
    return nnet

def calculate_act(weights, inputs):
    act = weights[-1] # initise with bias
    for i in range(len(weights)-1):
        act = act + weights[i] * inputs[i]
    return act

# transfer using sigmoid
def transfer(x):
    return 1.0/(1.0 + exp(-x))

# transfer derivative of sigmoid
def transfer_derivative(output):
    return output * (1.0-output)

def forward_pass(nnet, inputs):
    for layer in nnet:
        new_inputs = []
        for node in layer:
            act = calculate_act(node['weights'], inputs)
            act_tran = transfer(act)
            node['output'] = act_tran
            new_inputs.append(act_tran)
        inputs = new_inputs
    return inputs

# Error backpropagation
# Error for each output node
# error = (output-target)xtransfer_derivative(output)
# Error for a node in the hidden layer
# error = (weight_k x error_j) x transfer_derivative(output)

def back_propagate_error(nnet, target):
    errors = list()
    # calculate error for output layer
    out_layer = nnet[len(nnet)-1]
    for j in range(len(out_layer)):
        node = out_layer[j]
        err = node['output']-target[j]
        errors.append(err)
        node['delta'] = err * transfer_derivative(node['output'])

    for i in reversed(range(len(nnet)-1)): #omit output layer
        layer = nnet[i]
        for j in range(len(layer)): # current node affects the error for next layer
            err = 0.0
            for node_next in nnet[i+1]: 
                err = err + node_next['weights'][j] * node_next['delta'] #***
            errors.append(err)    
            node = layer[j]
            node['delta'] = err * transfer_derivative(node['output'])

# Update network weights with error
def update_weights(nnet, row, l_rate):
	for i in range(len(nnet)):
		inputs = row[:-1]
		if i != 0:
			inputs = [node['output'] for node in nnet[i - 1]]
		for node in nnet[i]:
			for j in range(len(inputs)):
				node['weights'][j] -= l_rate * node['delta'] * inputs[j]
			node['weights'][-1] -= l_rate * node['delta']

# Train a network for a fixed number of epochs
def train_network(nnet, train_data, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        total_err = 0.0
        for row in train_data:
            outputs = forward_pass(nnet, row)
            target = [0 for i in range(n_outputs)]
            target[row[-1]] = 1
            for i in range(len(outputs)):
                total_err += (target[i]-outputs[i])**2
            back_propagate_error(nnet, target)
            update_weights(nnet, row, l_rate)
        
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, total_err))

dataset = [[5.1,3.5,0],
[4.9,3.0,0],
[4.7,3.2,0],
[4.6,3.1,0],
[7.0,3.2,1],
[6.4,3.2,1],
[6.9,3.1,1],
[5.5,2.3,1],
[6.3,3.3,2],
[5.8,2.7,2],
[7.1,3.0,2],
[6.3,2.9,2]]

#Step 1 initialise a network
n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
seed(1)
nnet = init_network(n_inputs, 2, n_outputs)

#Step 2 train a network
train_network(nnet, dataset, 0.2, 20, n_outputs)
for layer in nnet:
    print(layer)