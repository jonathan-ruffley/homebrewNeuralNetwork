#a home brew neural network
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import math as m

#layer creation
class generateLayer():
    def __init__(self, nNeuronsAnterior, nNeurons, activationFunction):
        self.activationFunction = activationFunction
        self.b = np.round(stats.truncnorm.rvs(-1, 1, loc=0, scale=1, size = nNeurons).reshape(1, nNeurons), 3)
        self.w = np.round(stats.truncnorm.rvs(-1, 1, loc=0, scale=1, size=nNeurons*nNeuronsAnterior).reshape(nNeuronsAnterior, nNeurons), 3)

        
#activation functions
#sigmoid should take input value and give the function output
sigmoida = lambda x: m.pow(m.e, x)/(m.pow(m.e, x) + 1)
b = lambda x: sigmoida(0)*(1 - sigmoida(0))
sigmoid = (sigmoida, b)

#rectified linear unit: relu
def relu( x ):
    if x <= 0:
        return 0
    elif x > 0:
        return x
    else:
        print('error in relu function')

def reluD( x ):
    if x <= 0:
        return 0
    elif x > 0:
        return 1
    else:
        print('error in relu derivative')

#simple network
neurons = [2, 4, 8, 1]
activationFunctions = [relu, relu, sigmoid]

model = []
for j in range(len(neurons)-1):
    x = generateLayer(neurons[j], neurons[j+1], activationFunctions[j])
    model.append(x)
print(model)
