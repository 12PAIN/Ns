import numpy as np
import random as rd

training_data = [
    [0,1],
    [1, 2],
    [2, 3]
]

weights = [0,0]
weights = np.array(weights)
h = 0.1


def neuron(input , w):
    
    return input * w[1] + w[0]

def losses_func(network, training_data, wieghts):
    
    value = 0
    
    for item in training_data:
        value += np.power(network(item[0], wieghts) - item[1],2)
    
    return value

def gradFMP(fx, network, training_data, point, e0):
    
    grad = []
    
    for i in range(0, point.__len__()):
        
        currentPoint = np.array(point)
        
        currentPoint[i] = currentPoint[i] + e0
        
        dfx = (fx(network, training_data, currentPoint) - fx(network, training_data, point))/e0
        grad.append(dfx)
    
    return np.array(grad)

def trainNetwork(network, losses_function, training_data, gradFunction, weights, h):
    
    for i in range(0, 100):
        grad = gradFunction(losses_function, network, training_data, weights, 1e-10)
        weights = weights - (grad * h)
        lose = losses_function(network, training_data, weights)
        
        if i < 5:
            i += 1
            print("W: ", weights)
            print("G: ", grad)
            
    print("W: ", weights)
    
    print(network(5, weights))
        
trainNetwork(neuron, losses_func, training_data, gradFMP, weights, 0.1)
    
#Уже при 10 итерациях ГС нейрон даёт погрешность всего в 1 десятку
#А при 20 итерациях, погрешность около 1ой сотки
#30 - погрешность в 1e-3
#50 - 1e-5
#100 - 1e-9

