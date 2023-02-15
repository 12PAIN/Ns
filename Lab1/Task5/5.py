import numpy as np

def gradFMP(point, e0, fx):
    
    grad = []
    
    for i in range(0, point.__len__()):
        
        currentPoint = np.array(point)
        
        currentPoint[i] = currentPoint[i] + e0
        dfx = (fx(currentPoint) - fx(point))/e0
        grad.append(dfx)
    
    return np.array(grad)



def simpleGradDown(a0, h, e0, e1, b1, b2, gradFunc, fx):
    gradTemp = []
    
    a = np.array(a0)
    at = np.array(a)
    
    h0 = h
    
    temp = 1
    i = 0
    print("a",i,"=",a)
    n = 0
    
    R = []
    M = []
    
    while (i < 10000):
        
        grad = gradFunc(at, e0, fx)
        
        if(temp == 1):
            R = np.array(grad) - grad
            M = np.array(grad) - grad
        
        M = b1*M + (1 - b1)*grad
        R = b2*R + (1- b2)*np.power(grad,2)
        
        a = at - (M/(np.sqrt(R + e1)))*h
        at = np.array(a)
        
        n+=1
            
        if(i < 3):
            
            print("a",i,"=",a)
            print("grad",i,"=",grad)
            
        i = i + 1            
            
        if(temp == 1):
            temp = 0
            gradTemp = np.array(grad)
        else: 
            testValues = np.absolute((gradTemp - grad))
            
            count = 0
            
            for k in range(0, testValues.size): 
                if(testValues[k] < e0): 
                    count+=1
                    
            if(count > (2/3)*a.size): return a
                    
            gradTemp = np.array(grad)
    
    return a

        
def fx(point):
    return +np.log(point[0]*point[0]+1)+np.sin(point[0])


a0 = [2.5*np.pi]

print(simpleGradDown(np.array(a0), 1.1, 1e-11, 1e-10, 0.9, 0.9, gradFMP, fx))