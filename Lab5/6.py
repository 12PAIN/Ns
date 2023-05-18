import numpy as np


# A = x*w1+w3
# B = x*w2+w4

# Pr(Y = 0) = e^A/(e^A + e^B) 
# Pr(Y = 1) = e^B/(e^A + e^B)



# LF(w) = -ln(Pr(x1)) - ln(Pr(x2)) - ln(Pr(x3))

#Pr(x1) = e^(-w1 + w3)/(e^(-w1 + w3) + e^(-w2 + w4))
#Pr(x2) = e^(w4)/(e^(w3) + e^(w4))
#Pr(x1) = e^(w1 + w3)/(e^(w1 + w3) + e^(w2 + w4))

# -ln( e^(...) / ( e^(...) + e^(...) ) = ln( ( e^(...) + e^(...) / e^(...) ) = ln (1 + e^(...))

# ln(Pr(x1)) = ln (1 + e^(-w2 + w4))
# ln(Pr(x2)) = ln (1 + e^(w3))
# ln(Pr(x1)) = ln (1 + e^(w2 + w4))

def losesFunc(w, x):
    
    return np.log( 1 + np.pow(x[0] * w[1] + w[3], np.e) ) + np.log( 1 + np.pow( x[1]* w[0] + w[2] , np.e) ) + np.log( 1 + np.pow(x[2] * w[1] + w[3], np.e))
