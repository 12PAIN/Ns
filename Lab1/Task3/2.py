import numpy as np

def difFunc(x):
    delta = 1e-5
    return (fx(x+delta)-fx(x))/delta

def simpleGradDown(a0, h, e0):
    dfxTemp = 0
    
    a = a0
    at = a0
    
    temp = 1
    i = 0
    print("a",i,"=",a)
    while True:
        
        a = at - difFunc(at)*h
        at = a
        
        
        
        if(temp == 1):
            temp = 0
            dfxTemp = difFunc(at)
        else: 
            if(np.absolute(dfxTemp - difFunc(at)) < e0): break
            else: dfxTemp = difFunc(at)
        i = i + 1
        
        if(i < 5): 
            print("a",i,"=",a)
    
    return a
        
def fx(x):
    return np.cos(x) + np.log(x)


print (simpleGradDown(5,0.1,1e-10))