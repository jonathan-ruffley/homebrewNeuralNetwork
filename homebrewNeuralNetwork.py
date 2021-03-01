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

#Y here is fake labels, True/False
Y = [0]*10 + [1]*10
shuffle(Y)
Y = np.array(Y).reshape(len(Y),1)
#print(mse(output[-1], Y)[0])

#backpropagation, gradient descent
learningRate = 0.01
invertedIndex = list(range(len(output)-1))
invertedIndex.reverse()
errorStorage = []

for layer in invertedIndex:
    #print(model[-1].b)
    #print(model[-1].w)
    tempStorage = output[layer+1][1]
    if layer == invertedIndex[0]:
        x = mse(tempStorage, Y)[1] * model[layer].activationFunction[1](tempStorage)
        errorStorage.append(x)
    else:
        x = errorStorage[-1] @ Wtemp * model[layer].activationFunction[1](tempStorage)
        errorStorage.append(x)
    Wtemp = model[layer].w.transpose()

    model[layer].b = model[layer].b - errorStorage[-1].mean() * learningRate
    model[layer].w = model[layer].w - (output[layer].T @ errorStorage[-1])

print('MSE: {0}'.format(str(mse(output[-1], Y)[0])))
print('Estimation {0}'.format(str(output[-1])))
