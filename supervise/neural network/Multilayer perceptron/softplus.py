from math import log1p, exp
from random import random, seed

# ฟังก์ชัน softplus
def softplus(x):
    return log1p(exp(x))

# สร้าง network 2–3–1
def init_network(n_in, n_hid, n_out):
    seed(1)
    network = list()
    # Hidden layer
    hid_layer = []
    for h in range(n_hid):
        # สุ่ม weights ให้เท่ากับ n_in + bias
        rand_w = [random() for _ in range(n_in + 1)]
        # rand_w[0] เชื่อมกับ input[0], rand_w[1] เชื่อมกับ input[1], rand_w[2] = bias
        hid_layer.append({'weights': rand_w})
    network.append(hid_layer)
    # Output layer
    out_layer = []
    for o in range(n_out):
        # สุ่ม weights ให้เท่ากับ n_hid + bias
        rand_w = [random() for _ in range(n_hid + 1)]
        # rand_w[0..2] เชื่อมกับ hidden outputs[0..2], rand_w[3] = bias
        out_layer.append({'weights': rand_w})
    network.append(out_layer)
    return network

# ฟังก์ชัน forward propagation
def forward_propagation(inputs, network):
    print(f"\nInputs: {inputs}\n")
    
    # --- 1. Hidden layer ---
    hidden_outputs = []
    for idx, node in enumerate(network[0]):
        w = node['weights']
        # คำนวณ z = bias + Σ (w_i * input_i)
        z = w[-1] + sum(w[i] * inputs[i] for i in range(len(inputs)))
        a = softplus(z)
        hidden_outputs.append(a)
        # พิมพ์รายละเอียด
        print(f"Hidden Node {idx}:")
        print(f"  weights: w0={w[0]:.6f} (→ input0), w1={w[1]:.6f} (→ input1), bias={w[2]:.6f}")
        print(f"  z = {w[2]:.6f} + {w[0]:.6f}·{inputs[0]} + {w[1]:.6f}·{inputs[1]} = {z:.6f}")
        print(f"  a = softplus(z) = ln(1+e^{z:.6f}) = {a:.6f}\n")
    
    # --- 2. Output layer ---
    print(f"Hidden outputs → {hidden_outputs}\n")
    # ลูป node output ตัวเดียว
    for idx, node in enumerate(network[1]):
        w = node['weights']
        # คำนวณ z อีกรอบ
        z = w[-1] + sum(w[i] * hidden_outputs[i] for i in range(len(hidden_outputs)))
        a = softplus(z)
        print(f"Output Node {idx}:")
        print(f"  weights: w0={w[0]:.6f} (→ H0), w1={w[1]:.6f} (→ H1), w2={w[2]:.6f} (→ H2), bias={w[3]:.6f}")
        print(f"  z = {w[3]:.6f} "
              f"+ {w[0]:.6f}·{hidden_outputs[0]:.6f} "
              f"+ {w[1]:.6f}·{hidden_outputs[1]:.6f} "
              f"+ {w[2]:.6f}·{hidden_outputs[2]:.6f} "
              f"= {z:.6f}")
        print(f"  a = softplus(z) = ln(1+e^{z:.6f}) = {a:.6f}\n")
    
    return a

# เรียกใช้
nnet = init_network(2, 3, 1)
output = forward_propagation([1, 3], nnet)
print(f"Final network output = {output:.6f}")
