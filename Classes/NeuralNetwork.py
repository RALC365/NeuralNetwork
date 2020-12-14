from Classes import NetLayer as ntl
from tabulate import tabulate

class NeuralNetwork:
    layers= []

    def __init__(self):
        self.layers= []
        
#Método Modificado
    def evaluar(self, X):
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i].neurons)):
                neuron = self.layers[i].neurons[j]
                in0 = 0.0
                if i==0:
                    #neuron[0]: bias, neuron[1]: array de pesos
                    in0 = (neuron[0] * neuron[1][0]) + (neuron[1][1]*X[0]) + (neuron[1][2]*X[1]) 
                else:
                    in0 = (neuron[0] * neuron[1][0])
                    #evaluamos la gn de las neurons de la capa enterior 
                    #-2
                    for k in range(len(self.layers[i-1].neurons)-2):
                        anterior = self.layers[i-1].neurons[k]
                        #print(neuron[1][k+1])
                        in0 = in0 + (neuron[1][k+1] * anterior[2])
                neuron[2] = self.layers[i].sigmoide(in0)


#Método Modificado
    def imprimir_salidas(self,vector):
        imp = []
        imp.append(['Vector', vector])
        for neuron in self.layers[len(self.layers)-1].neurons:
            imp.append(["g(n)",neuron[2]])
        print(tabulate(imp,  headers="firstrow",tablefmt='fancy_grid'))
        
#Método Modificado
    # se generan las capas con sus neuronas y los pesos
    def generar(self, entradas, layers,neurons):
        #se agregan las capas y
        for i in range(layers):
            layer = ntl.NetLayer()
            self.layers.append(layer)
            for j in range(0,neurons):
                print("Valor nue: " + str((len(self.layers[i-1].neurons))))
                self.layers[i].addNeuron(entradas+1)
                #print("{:.2}".format(self.layers[i].neurons[j].pesos[0]))
                #print("{:.2}".format(self.layers[i].neurons[j].pesos[1]))
                #print("{:.2}".format(self.layers[i].neurons[j].pesos[2]))
                #print("")

                if i>0:
                    #print("cuanto? ",len(self.layers[i-1].neurons))
                    entradas = len(self.layers[i-1].neurons)
                    print("Valor: " + str((entradas)))
