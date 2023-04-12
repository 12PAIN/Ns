import numpy as np
import random as rd

def fAct(input):
    return max(0, input)

def network(x, w, f):
    return f(f(w[0][0]*x[0] + w[1][0]*x[1]) + f(w[0][1]*x[0] + w[1][1]*x[1]))

def losses_func(network, dataset, w, fActivation):
    
    value = 0
    
    for item in dataset:
        inputData = [item[0], item[1]]
        value += np.power(network(inputData, w, fActivation) - item[2],2)

    return value

def gradFMP(lFx, network, dataset, w, fActivation, e0):
    
    grad = list()
    
    lenY = 0
    
    for item in w:
        lenY += 1
    
    for i in range(0, lenY):
        currentPoint = np.array(w)
        grad.append(list())
        for j in range(0, len(w[i])):
            currentPoint[i][j] = currentPoint[i][j] + e0

            dfx = (lFx(network, dataset, currentPoint, fActivation) - lFx(network, dataset, w, fActivation))/e0
            
            grad[i].append(dfx)
    
    return np.array(grad)

def trainNetwork(network, losses_function, dataset, gradFunction, weights, fActivation, h):
    
    for i in range(0, 100):
                   
        grad = gradFunction(losses_function, network, dataset, weights, fActivation, 1e-10)
        weights = weights - (grad * h)
        
        if i < 2:
           print("W:",i," \n", weights)
           print("G:",i," \n", grad)
            
    print("W: ", weights)
    
    print(network([1, 0], weights, fAct))
    
dataset = [
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 1]
]

weights = [
    [0.5, -0.5],
    [-0.5, 0.5]
]

trainNetwork(network, losses_func, dataset, gradFMP, weights, fAct, 0.1)

