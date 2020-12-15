from random import uniform
import numpy as np

class NetLayer:
    #[pesos, bias, g(n), gradiente]
    neurons= []

    def __init__(self):
        self.neurons= []

    #función de activición
    def sigmoide(self, x):
        return 1.0/(1.0 + np.exp(-x))

    #Agregar neurona [bias, pesos, g(n), delta]
    def addNeuron(self, cNeuron):
        w = []
        for i in range(cNeuron):
            w.append(uniform(-1,1))
        self.neurons.append([1,w,0,0])
        
        