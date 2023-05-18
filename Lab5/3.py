import numpy as np

def func(p1, p2):
    return -1*np.log(p1) - np.log(p2)

loseFuncValue = func(0.5,0.5)

currentLoseValue = loseFuncValue/1.5

p00 = 0.6

#### -ln(p11) - ln(p00) = losesFunc => ln(p11) + ln(p00) = -losesFunc 
#### => ln(p11 * p00) = ln(e^(-losesFunc))
#### => p11 * p00 = e^-losesFunc
#### => p11 = (e^-losesFunc)/p00

p00 = 0.6

p11 = (np.power(np.e, -1*currentLoseValue))/p00

print(p11)