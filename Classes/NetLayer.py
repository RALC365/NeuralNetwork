from random import uniform
import numpy as np

#CLASE LISTA
# representar una capa de la red (oculta o de salida)
class NetLayer:
    #[pesos, bias, g(n), gradiente]
    neurons= []

    def __init__(self):
        self.neuronas= []

    #función de activición
    def sigmoide(self, x):
        return 1.0/(1.0 + np.exp(-x))

    #Agregar neurona [bias, pesos, g(n), delta]
    def addNeuron(self, cantidad):
        w = []
        for i in range(cantidad):
            w.append(uniform(-1,1))
        self.neurons.append([1,w,0,0])
        
        