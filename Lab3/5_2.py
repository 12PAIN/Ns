import numpy as np

dataset = [
    [50, 0, 1, 0],
    [60, 1, 2, 1],
    [80, 1, 3, 0],
    [100, 0, 4,1]
]

def avg(list, col):
    value = 0
    for i in range(0, len(list)):
        value += list[i][col]
    
    return value/len(list)

def s(list, col):
    avgValue = avg(list, col)
    
    value = 0
    
    for i in range(0, len(list)):
        value += np.power((list[i][col] - avgValue), 2)
    
    return np.sqrt(value/(len(list)-1))

avgWeight = avg(dataset, 0)
avgGroup = avg(dataset, 2)

sWeight = s(dataset, 0)
sGroup = s(dataset, 2)

def normalize(obj):
    
    weight = (obj[1] - avgWeight)/sWeight
    group = (obj[3] - avgGroup)/sGroup
    
    
    newObj = [obj[0], weight, obj[2], group, "?"]
    
    return newObj

obj = ["A", 90, 1 , 1]

print(normalize(obj))