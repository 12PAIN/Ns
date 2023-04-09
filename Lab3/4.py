import numpy as np
import random as rd

training_data = [
    [-1,1],
    [0, 0],
    [1,-1]
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
        
        rd.shuffle(training_data)
        
        for j in range (0, 1):
            
            data = training_data[j:j+1]
            
            grad = gradFunction(losses_function, network, data, weights, 1e-10)
            weights = weights - (grad * h)
        
        if i < 3:
            i += 1
            print("W: ", weights)
            print("G: ", grad)
            
    print("W: ", weights)
    
    print(network(5, weights))
        
trainNetwork(neuron, losses_func, training_data, gradFMP, weights, 0.1)
    
#Уже при 10 - 6 10ых погрешность
# 20 - 2 10
# 30 - 8 соты
# 40 - 3 сотых
# 60 - 5 тысячных
# 100 - 3e-7

