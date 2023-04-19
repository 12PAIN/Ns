import numpy as np
import random as rd

def fAct(input):
    return max(0, input)

def network(x, w, f):
    return f(f(w[0][0]*x[0] + w[1][0]*x[1]) + f(w[0][1]*x[0] + w[1][1]*x[1]))

def losses_func(network, dataset, w, fActivation, C):
    
    value = 0
    weightsSum = 0
    
    for item in dataset:
        inputData = [item[0], item[1]]
        value += np.power(network(inputData, w, fActivation) - item[2],2)

    if C != 0:
        for col in w:
            for weight in col:
                weightsSum += np.power(weight, 2)
    
    value += C * weightsSum

    return value

def gradFMP(lFx, network, dataset, w, fActivation, e0, C):
    
    grad = []
    
    for i in range(0, len(w)):
        
        grad.append(list())
        for j in range(0, len(w[i])):
            currentPoint = np.array(w)
            currentPoint[i][j] = currentPoint[i][j] + e0

            dfx = (lFx(network, dataset, currentPoint, fActivation, C) - lFx(network, dataset, w, fActivation, C))/e0
            
            grad[i].append(dfx)
    
    return np.array(grad)

def trainNetwork(network, losses_function, dataset, gradFunction, weights, fActivation, h, C):
    
    for i in range(0, 100):
                   
        grad = gradFunction(losses_function, network, dataset, weights, fActivation, 1e-10, C)
        weights = weights - (grad * h)
        
        if i < 2:
           print("W:",i," \n", weights)
           print("G:",i," \n", grad)
            
    print("W: ", weights)
    
    print(network([0, 0], weights, fAct))
    print(network([0, 1], weights, fAct))
    print(network([1, 0], weights, fAct))
    print(network([1, 1], weights, fAct))
    print("h=", h, "C= ", C)
    
dataset = [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
]

weights = [
    [0.5, -0.5],
    [-0.5, 0.5]
]

h = 0.1
с = 1

trainNetwork(network, losses_func, dataset, gradFMP, weights, fAct, h, с)

