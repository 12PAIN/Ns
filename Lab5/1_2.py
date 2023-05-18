import numpy as np

def func(p1, p2):
    return -1*np.log(p1) - np.log(p2)

loseFuncValue = func(0.5,0.5)

print(loseFuncValue)

#### -ln(p11) - ln(p00) = losesFunc => ln(p11) + ln(p00) = -losesFunc 
#### => ln(p11 * p00) = ln(e^(-losesFunc))
#### => p11 * p00 = e^-losesFunc
#### => p11 = (e^-losesFunc)/p00

p00 = 0.6
p01 = 0.4

p11 = (np.power(np.e, -1*loseFuncValue))/p00

#### p01 = 1-p11
p01 = 1 - p11

print(p11)
print(p01)

#testing
print(func(0.5,0.5))
print(func(p00, p11))