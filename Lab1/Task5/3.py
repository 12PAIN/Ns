import numpy as np


def gradFMP(point, e0, fx):
   
    grad = []
   
    for i in range(0, point.__len__()):
       
        currentPoint = np.array(point)
       
        currentPoint[i] = currentPoint[i] + e0
        dfx = (fx(currentPoint) - fx(point))/e0
        grad.append(dfx)
   
    return np.array(grad)



def simpleGradDown(a0, h, e0, T, gradFunc, fx):
    gradTemp = []
   
    a = np.array(a0)
    at = np.array(a)
   
    h0 = h
   
    temp = 1
    i = 0
    print("a",i,"=",a)
    n = 0
    while (n < T):
       
        grad = gradFunc(at, e0, fx)
        a = at - grad*h
        at = np.array(a)
       
        n+=1
        h = h0 * np.power(np.e, (-1*n)/T)
           
        if(i < 10):
            i = i + 1
            print("a",i,"=",a)
            print("grad",i,"=",grad)
           
        if(temp == 1):
            temp = 0
            gradTemp = np.array(grad)
        else:
            testValues = np.absolute((gradTemp - grad))
            for k in range(0, testValues.size):
                if(testValues[k] < e0):
                    return a
            else: gradTemp = np.array(grad)
   
    return a


       
def fx(point):
    return np.cos(point[0])+np.log(point[0])


a0 = [5.0]


print(simpleGradDown(np.array(a0), 0.1, 1e-10, 100, gradFMP, fx))