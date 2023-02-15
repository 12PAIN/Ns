import numpy as np

def difFunc(x, delta, fx):

    return (fx(x+delta)-fx(x))/delta

def difFunc2(x, delta, fx):

    answer = (fx(x-2*delta) - 8*fx(x-delta) + 8*fx(x + delta) - fx(x + 2*delta))/(12*delta)
        
    return answer

def simpleGradDown(a0, h, e0, difFunc, fx):
    dfxTemp = 0
    
    a = a0
    at = a
    
    temp = 1
    i = 0
    print("a",i,"=",a)
    while True:
        
        dfx = difFunc(at, e0, fx)
        a = at - dfx*h
        at = a
            
        if(i < 100):
            i = i + 1
            print("a",i,"=",a)
            print("dfx",i,"=",dfx)
            
        if(temp == 1):
            temp = 0
            dfxTemp = dfx
        else: 
            if(np.absolute(dfxTemp - dfx) < e0): break
            else: dfxTemp = dfx
    
    return a
        
def fx(x):
    return x*(x*x-9)


print(simpleGradDown(-2,0.1,1e-10, difFunc, fx))
print()
print()
print(simpleGradDown(-2,0.1,1e-10, difFunc2, fx))
#print(difFunc(-2, 1e-10, fx))
#print(difFunc2(-2, 1e-10, fx))