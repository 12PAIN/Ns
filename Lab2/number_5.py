import numpy as np

print("Введите x:")
x = float(input())

print("Выберите функцию активации:")
print("1. Сигмоида")
print("2. ReLU")
print("3. Гиперболический тангенс")

choice = int(input())

def sigmoida(z):
    return 1.0/(1.0+(np.power(np.e, -1.0*z)))

def relu(z):
    if z <= 0: return 0
    else: return z
    
def tanh(z):
    return 2.0/(1.0+(np.power(np.e, -2.0* z)))

def neuron(x, func):
    
    w1 = 1
    w0 = 1

    return func(x*w1 + w0)

if choice == 1: print(neuron(x, sigmoida))
if choice == 2: print(neuron(x, relu))
if choice == 3: print(neuron(x, tanh))