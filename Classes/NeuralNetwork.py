from Classes import NetLayer as ntl
from tabulate import tabulate

class NeuralNetwork:
    layers= []

    def __init__(self):
        self.layers= []
        
    def evaluar(self,vector):
        ban = 0
        before = ntl.NetLayer()
        for layer in self.layers:
            for neuron in layer.neurons:
                x = 0.0
                if ban != 0:
                    x = neuron[0] * neuron[1][0]
                    for i in range(len(before.neurons)):
                        x += neuron[1][i+1] * before.neurons[i][2]
                else:
                    x = neuron[0] * neuron[1][0]
                    for i in range(len(vector)):
                        x += neuron[1][i+1] * vector[i]
                neuron[2] = layer.sigmoide(x)
            ban = 1
            before = layer

    def printG(self,vector):
        imp = []
        imp.append(['Vector', vector])
        for neuron in self.layers[-1].neurons:
            imp.append(["g(n)",neuron[2]])
        print(tabulate(imp,  headers="firstrow",tablefmt='fancy_grid'))
        
    def neuRandom(self, cIn, cLayers,cNeurons):
        for i in range(cLayers):
            layer = ntl.NetLayer()
            self.layers.append(layer)
            for j in range(0,cNeurons):
                self.layers[i].addNeuron(cIn+1)
                if i>0:
                    cIn = len(self.layers[i-1].neurons)
        

    #Métodos Ejercicio 3
    def AddHiddenLayer(self, cNeurons, cIn):
        layer = ntl.NetLayer()
        for i in range(cNeurons):
            layer.addNeuron(cIn+1)
        print("Agregando Capa oculta...")
        self.layers.append(layer)
        
    def AddOutLayer(self, cNeurons):
        layer = ntl.NetLayer()
        cIn = len(self.layers[len(self.layers)-1].neurons)+1
        for i in range(cNeurons):
            layer.addNeuron(cIn)
        print("Agregando capa de salida...")
        self.layers.append(layer)