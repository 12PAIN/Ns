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
    
    value += np.power(network(training_data[0], wieghts) - training_data[1],2)
    
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
        rd.shuffle(training_data)
        for j in range(0, 2):
            
            grad = gradFunction(losses_function, network, training_data[j], weights, 1e-10)
            weights = weights - (grad * h)
        
        if i < 3:
            i += 1
            print("W: ", weights)
            print("G: ", grad)
            
    print("W: ", weights)
    
    print(network(5, weights))
        
trainNetwork(neuron, losses_func, training_data, gradFMP, weights, 0.1)

# 20 Итераций - ошибка на 3 сотки
# 30 - на 1 сотку
# 60 - 1e-10
# 100 - 7e-7