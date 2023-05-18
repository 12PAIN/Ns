import numpy as np

Weights = []
Biases = []

def initializeNetwork(inputsCount, outputsCount, Layers, neurons, weights, biases):
    
    MaxLayerCount = (int)(max(
        
        max(np.ceil(np.sqrt(inputsCount*neurons)),
            np.ceil(np.sqrt(inputsCount*inputsCount))),
        
        max(
            max(np.ceil(np.sqrt(outputsCount*neurons)),
            np.ceil(np.sqrt(outputsCount*outputsCount))),
            np.ceil(np.sqrt(neurons*neurons)))))
    
    print(MaxLayerCount)
    
    weights.append([])
    
    #Inputs -> First Layer
    
    for i in range(0, MaxLayerCount):
        weights[0].append([])
        for j in range(0, MaxLayerCount):
            
            if i < inputsCount:
                #weights[i][j].append(getInitWeight())
                weights[0][i].append(1)
            else:
                weights[0][i].append(0)
            
    # Inner Layers -> Inner Layers
    
    for Layer in range(1, Layers):
        weights.append([])
        for i in range(0, neurons):
            weights[Layer].append([])
            for j in range(0, neurons):
                #weights[i][j].append(getInitWeight())
                weights[Layer][i].append(1)
                
    # Biases
    
    for layer in range(0, Layers):
        biases.append([])
        for i in range(0, neurons):
            #weights[i][j].append(getInitWeight())
            biases[layer].append(1)
           
    #Last inner Layer -> Output Layer
    
    weights.append([])
    
    for i in range(0, MaxLayerCount):
        weights[Layers].append([])
        for j in range(0, MaxLayerCount):
            
            if j < outputsCount:
                #weights[i][j].append(getInitWeight())
                weights[Layers][i].append(1)
            else:
                weights[Layers][i].append(0)
    

def net(inputs, weights, biases, outputs):
    tempInp = np.array(inputs)
    tempBiases = np.array(biases)
    
    value = 
    
    for i in range(0, len(Weights)):
        value = tempInp * weights[i] + tempBiases[i]
    
    outputs = value
    return value

def getInitWeight():
    
    pass

def network(inputs):
    pass

inputsCount = 2
outputCount = 1
Layers = 1
Neurons = 2

initializeNetwork(inputsCount, outputCount, Layers, Neurons, Weights, Biases)

input = [0, 0]
output = [0, 0]

print(net(input, Weights, Biases, output))