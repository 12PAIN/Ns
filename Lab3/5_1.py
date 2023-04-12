
dataset = [
    [50, 0, 1, 0],
    [60, 1, 2, 1],
    [80, 1, 3, 0],
    [100, 0, 4,1]
]

def max(list, col):
    
    maxValue = -1
    
    for i in range(0, len(list)):
        
        if list[i][col] > maxValue: maxValue = list[i][col]
    
    return maxValue

def min(list, col):
    
    maxValue = 1000000
    
    for i in range(0, len(list)):
        
        if list[i][col] < maxValue: maxValue = list[i][col]
    
    return maxValue

maxWeigth = max(dataset, 0)
minWeight = min(dataset, 0)

maxGroup = max(dataset, 2)
minGroup = min(dataset, 2)

def normalize(obj):
    
    weight = (obj[1] - minWeight)/(maxWeigth - minWeight)
    group = (obj[3] - minGroup)/(maxGroup - minGroup)
    
    
    newObj = [obj[0], weight, obj[2], group, "?"]
    
    return newObj

obj = ["A", 90, 1 , 1]

print(normalize(obj))