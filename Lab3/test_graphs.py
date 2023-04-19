import numpy as np

networkStruct = [
    [0,0,1,1,0,0],
    [0,0,1,1,0,0],
    [0,0,0,0,1,0],
    [0,0,0,0,1,0],
    [0,0,0,0,0,1],
    [0,0,0,0,0,0]
]


def fAct(input):
    return max(0, input)

def network(net, inputVector, fAct):
    
    inputsCount = initializeInput(net, inputVector)
    
    for i in range(0, len(net)):
        for j in range(0, len(net[i])):
            if (net[i][j] != 0) and ( i != j) and (i < inputsCount):
                net[i][j] = net[i][j] * net[i][i]
            
            if (i == j) and (i >= inputsCount):
                value = 0
                
                for k in range(0, len(net)):
                    if (net[k][j] != 0) and (k != j):
                        value += net[i][k]
                
                net[i][j] += value
                net[i][j] = fAct(value)
                
                for k in range(0, len(net[i])):
                    if (net[i][k] != 0) and (i != k):
                        net[i][k] = net[i][k] * net[i][j]

def initializeInput(net, inputVector):
    
    inputsCount = len(inputVector)
    
    for i in range(0, inputsCount):
        net[i][i] = inputVector[i]
            
    return inputsCount
    
def initWeigths(net):
    pass

def initBiases(net):
    pass    

def createFullyConnectedNetwork(inputCount, innerLayersCounts, outputCount):
    
    net = []
    
    neuronsCount = inputCount + sum(innerLayersCounts) + outputCount
    
    for i in range(0, inputCount):
        net.append([])
        for j in range(0, neuronsCount):

                if j > i and j < innerLayersCounts[0]:
                    net.append(1)
                
                else:
                    net.append(0)

    innerLayerCounter = 0
    
    for layer in innerLayersCounts:
        if innerLayerCounter != 0:
            
            left = inputCount-1 + sum(innerLayersCounts[0:innerLayerCounter-1])
            innerLayerCounter += 1
            
            for i in range(left, left + layer[i]):
                for j in range(0, neuronsCount):
                    
                
            
                
    
    return net