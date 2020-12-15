from Classes import NetLayer as ntl
from Classes import NeuralNetwork as nn

import sys
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

# ignore all future warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

def BackPropagation(datat):
    nets = []
    data = {} #para archivo csv
    #se generan 20 redes 
    for i in range(20):
        net = nn.NeuralNetwork()
        net.neuRandom(2,2,2)
        nets.append(net)
    ins = datat.iloc[:,0:2].values
    outs = datat.iloc[:,2:4].values
    
    #evaluamos cada red por los 4 vectores de entrada
    print("Please wait...")
    for i in range(len(ins)):
        vector = ins[i]
        redColumna = []
        mseMax = []
        mseMin = []
        mseProm = []
        for net in nets:
            redColumna.append(nets.index(net)+1)
            errorA =1
            MSE = []
            #rondas
            for j in range(100): #rondas 100
                net.evaluar(vector)
                calculoGradiente(net,vector)
                cambiarPesos(net,vector)
                #MSE de la ronda
                ultimasN = net.layers[len(net.layers)-1].neurons
                error = (np.square(outs[i] - [ultimasN[0][2],ultimasN[1][2]])).mean()

                #Vemos si el error ha aumentado para detener el proceso
                if error >= errorA: break
                #Si no, continuamos
                errorA = error
                MSE.append(errorA)
            
            mseMin.append(np.amin(MSE))
            mseMax.append(np.amax(MSE))
            mseProm.append(np.mean(MSE))

            #guardamos los resultados de la red
            save(net,2,'BackPropagation_Results/JSON/results_net_'+str(nets.index(net)+1)+'.json')

        data = {'ID': redColumna,
            'MSE MIN': mseMin,
            'MSE_MAX': mseMax,
            'MSE_PROM': mseProm}
        archivo = pd.DataFrame(data, columns=['ID','MSE_MIN','MSE_MAX','MSE_PROM'])
        archivo.to_csv('BackPropagation_Results/CSV/inputs_net_'+str(vector[0])+str(vector[1])+'.csv')

        #Gráficos
        #Seteamos los labels de las lineas
        plt.plot(mseMax,  label = "Max")
        plt.plot(mseProm,  label = "Prom")
        plt.plot(mseMin,  label = "Min")
        plt.legend()
        #Agregamos las etiqueta y títulos
        plt.title(vector, fontsize=10)
        plt.xlabel("Neural Network", fontsize=10)
        plt.ylabel("Error", fontsize=10)
        #Guardamos el gráfico
        plt.savefig("BackPropagation_Results/Graphics/Results_"+str(vector[0])+str(vector[1])+".png")
        #Limiamos el plot para el próximo gráfico
        plt.clf()

    #todavia no tengo claro cuantas graficas quieren exactamente
    #o cuales valores quiere (por epoca no solo se obtenie un error????)
    print("Pesos guardados en 'BackPropagation_Results/JSON/'")
    print("CSV guardado en en 'BackPropagation_Results/CSV/'")
    print("Gráficos guardados en 'BackPropagation_Results/Graphics/'")

#metodo para calcular el gradiente de las neuronas
def calculoGradiente(net,vector):
    for i in range (len(net.layers)):
        indice = (len(net.layers)-1)-i
        capa =net.layers[indice]
        for j in range(len(capa.neurons)):
            neurona = capa.neurons[j]
            valor = 0
            gp = neurona[2]*(1-neurona[2])
            if indice ==(len(net.layers)-1):
                #estoy en la salida
                valor = vector[j]-neurona[2]
            else:
                #capa oculta
                anteriores = net.layers[indice+1].neurons
                for nAnterior in anteriores:
                    valor = valor + (nAnterior[1][0]*nAnterior[3])

            neurona[3] = gp*valor 
            
#metodo para cambiar los pesos de la net
def cambiarPesos(net, vector):
    alfa = 0.1
    a0 = 1     
    for i in range(len(net.layers)):
        capa = net.layers[i]
        for neurona in capa.neurons:
            #print("-> <-")
            for j in range(len(neurona[1])):
                if j==0:
                    neurona[1][j]  = neurona[1][j] + alfa *  a0 * neurona[3]
            
                if i >0:
                    if j>0:
                        anteriorN = net.layers[i-1].neurons
                        neurona[1][j]  = neurona[1][j] + alfa *  anteriorN[j-1][3] * neurona[3]
                else:
                    if j>0:
                        valorV = vector[j-1]
                        neurona[1][j]  = neurona[1][j] + alfa *  valorV * neurona[3]
                  
                #print(neurona.pesos[j], end=",   ")

#Método modificado
#metodo para cargar datos los datos del archivo JSON
def load(file):
    neuralNet = nn.NeuralNetwork()
    with open(file) as oFile:
        jsonF = json.load(oFile)
        for layer in jsonF['capas']:
            nLayer = ntl.NetLayer()
            for neuron in layer['neuronas']:
                print(neuron['pesos'])
                w = neuron['pesos']
                nLayer.neurons.append([1,w,0,0])
            neuralNet.layers = np.append(neuralNet.layers, nLayer)

    for i in range(0,2):
        for j in range(0,2):
            vector= [i,j]
            neuralNet.evaluar(vector)
            neuralNet.printG(vector)


#metodo cambiado
def save(neuNet, ins, fileO):
    jsonF ={}
    jsonF['entradas'] = ins
    jsonF['capas'] = []
    for i in range(len(neuNet.layers)):
        jsonN = {}
        jsonN['neuronas'] = []
        for j in range(len(neuNet.layers[i].neurons)):
            neuron = neuNet.layers[i].neurons[j]
            w = {}
            w['pesos'] = neuron[1]
            jsonN['neuronas'].append(w)
        jsonF['capas'].append(jsonN) 
    with open(fileO, 'w') as file:
        json.dump(jsonF, file)


#solo inicia si es el proceso inicial#
if __name__ == "__main__":
    #python t4_back_propagation.py part2_train_data.csv
    file = ''
    if len(sys.argv) == 1:
        file = './Datos/part2_train_data.csv'
    else:
        file = sys.argv[1]

    data = pd.read_csv(file, engine='python')

    BackPropagation(data)

    
