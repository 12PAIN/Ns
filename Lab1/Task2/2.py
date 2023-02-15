import numpy as np

a0_1 = 0.6
h1 = 0.1

e0=1e-4

a0 = a0_1
a = a0

fx0 = -1e10
fx1 = 4*a0*a0*a0

temp = 0

while(np.absolute(fx1-fx0) > e0):
    a = a0 - fx1*h1
    if(temp == 0): 
        print("1 Шаг x^4:",fx1*h1)
        temp = 1
    a0 = a
    fx0 = fx1
    fx1 = 4*a*a*a

    
    
print(a)
    