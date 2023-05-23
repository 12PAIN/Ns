import numpy as np


def losesFunc(w, x):
    return np.log( 1 + np.exp(x[0] * w[1] + w[3]) ) + np.log( 1 + np.exp( x[1] * w[0] + w[2]) )
def softmax(output, index):
    return np.exp(output[index]) / (np.exp(output[index]) + np.exp(output[abs(index - 1)]))

def network(input, w):
    return [input * w[0] + w[2], input * w[1] + w[3]]

def gradFMP(lFx, dataset, w, e0):
    
    grad = []
    
    for i in range(0, len(w)):
        
        currentWeight = np.array(w)
        currentWeight[i] = currentWeight[i] + e0

        dfx = (lFx(w, dataset) - lFx(currentWeight, dataset))/e0
        
        grad.append(dfx)
    
    return np.array(grad)

def train(dataset, w, lFx, h, e0):
    
    epoches = 20
    
    for i in range(0, epoches):
        grad = gradFMP(lFx, dataset, w, e0)
        w = w + grad*h

    print("x1 Pr0:" ,softmax(network(dataset[0], w), 0))
    print("x1 Pr1:" ,softmax(network(dataset[0], w), 1))
    
    print("x2 Pr0:" ,softmax(network(dataset[1], w), 0))
    print("x2 Pr1:" ,softmax(network(dataset[1], w), 1))
    
    print("lFx = ", losesFunc(w, dataset))
    print("stupid classifier:", -1*np.log(1/2) - 1  * np.log(2/2))
    
weights = np.array([0,0,0,0])

dataset = [-1,1]

train(dataset, weights, losesFunc, 0.1, 1e-4)


