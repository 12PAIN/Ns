import numpy as np


#Дураций классификатор будет работать так:
#Вероятность для класса 0 будет равна 3/4
#Вероятность для класса 1 будет равна 1/4

#Ф-ция потерь дурацкого классификатора
def loseFuncStupidClassificatorBinary(m0, m1):
    
    value = -1*m0
    value *= np.log(m0/(m0+m1))
    
    value -= m1*np.log(m1/(m0+m1))
    
    return value

#Она будет равна:
print(loseFuncStupidClassificatorBinary(3,1))

#Теперь для классификатора K2:
#Нам известна вероятность принадлежности классу 0 p0=0.6
p0 = 0.6

#Тк функции потерь равны, то вероятность принадлежности классу p1 будет:
#### -ln(p1) - 3*ln(p0) = losesFunc => ln(p1) + 3*ln(p0) = -losesFunc 
#### => ln(p1 * p0^3) = ln(e^(-losesFunc))
#### => p1 * p0^3 = e^-losesFunc
#### => p1 = (e^-losesFunc)/p0^3

p1 = (np.power(np.e, -1*loseFuncStupidClassificatorBinary(3,1)))/np.power(p0,3)
print(p1)

#Проверим
def func(p1, p2, p3, p4):
    return -1*np.log(p1) - np.log(p2) -np.log(p3) -np.log(p4)

print(func(p0,p0,p0,p1))
#Равны))