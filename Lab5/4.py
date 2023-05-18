import numpy as np

m1=10
m2=20

m01 = 5
m11 = 5
m02 = 10
m12 = 10

def loseFuncStupidClassificatorBinary(m0, m1):
    
    value = -1*m0
    value *= np.log(m0/(m0+m1))
    
    value -= m1*np.log(m1/(m0+m1))
    
    return value

print(loseFuncStupidClassificatorBinary(m01, m11))
print(loseFuncStupidClassificatorBinary(m02, m12))