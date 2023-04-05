import numpy as np

print("Введите x:")
x = float(input())

def relu(z):
    if z <= 0: return 0
    else: return z
    
def neuron_1_input(x, w1, w0, func):
    return func(x*w1 + w0)

def neuron_2_inputs(x, y, w1, w2, w0, func):
    return func(x*w1 + y*w2 + w0)

first = neuron_1_input(x, 1.0, 1.0, relu)
second = neuron_1_input(x, 2.0, -5.0, relu)

value = neuron_2_inputs(first, second, -2.0, 4.0, 10.0, relu)

print(value)