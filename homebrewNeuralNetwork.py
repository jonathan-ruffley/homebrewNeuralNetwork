#a home brew neural network
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import math as m
from random import shuffle

#layer creation
class generateLayer():
    def __init__(self, nNeuronsAnterior, nNeurons, activationFunction):
        self.activationFunction = activationFunction
        self.b = np.round(stats.truncnorm.rvs(-1, 1, loc=0, scale=1, size = nNeurons).reshape(1, nNeurons), 3)
        self.w = np.round(stats.truncnorm.rvs(-1, 1, loc=0, scale=1, size=nNeurons*nNeuronsAnterior).reshape(nNeuronsAnterior, nNeurons), 3)

def mse(Yhat, Ytrue):
    x = (np.array(Yhat) - np.array(Ytrue)) **2
    x = np.mean(x)
    y = np.array(Yhat) - np.array(Ytrue)
    return(x, y)
    
#activation functions
#sigmoid should take input value and give the function output
sigmoid = (
    lambda x: 1 / (1 + np.exp(-x)),
    lambda x: (1/(1 + np.exp(-x))) * (1 - (1/(1 + np.exp(-x))))
    )

#rectified linear unit: relu
def derivativeRelu( x ):
    x[x<=0] = 0
    x[x>0] = 1
    return x

relu = (
    lambda x: x * (x > 0),
    lambda x: derivativeRelu(x)
    )

#simple network
neurons = [2, 4, 8, 1]
activationFunctions = [relu, relu, sigmoid]

model = []
for j in range(len(neurons)-1):
    x = generateLayer(neurons[j], neurons[j+1], activationFunctions[j])
    model.append(x)

output = [np.round(np.random.randn(20,2),3)]

#now try to do a prediction
for k in range(len(model)):
    z = output[-1] @ model[k].w + model[k].b
    act = model[k].activationFunction[0](z)
    output.append(act)
#print(output[-1])

Y = [0]*10 + [1]*10
shuffle(Y)
Y = np.array(Y).reshape(len(Y),1)
print(mse(output[-1], Y)[0])

