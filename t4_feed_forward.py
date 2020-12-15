from Classes import NetLayer as ntl
from Classes import NeuralNetwork as nn

import sys
import numpy as np
from random import uniform
import json
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

#Método modificado
#metodo para cargar datos los datos del archivo JSON
def load(file):
    #instanciamos la red
    neuralNet = nn.NeuralNetwork()
    with open(file) as oFile:
        #leemos el archivo en formato JSON
        jsonF = json.load(oFile)
        #Iteramos por capas y neuronas
        for layer in jsonF['capas']:
            nLayer = ntl.NetLayer()
            for neuron in layer['neuronas']:
                w = neuron['pesos']
                #Añadimos la neurona como arreglos [baias, arregloPesos, g(n), delta)
                nLayer.neurons.append([1,w,0,0])
            #Añadimos la capa con las neuronas a la red
            neuralNet.layers = np.append(neuralNet.layers, nLayer)
        return neuralNet


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
    #python t4_feed_forward.py part1.red.prueba.json part2_train_data.csv

    file = ''
    if len(sys.argv) == 1:
        file = './Datos/part1_red_prueba.json'
    else:
        file = sys.argv[1]

    #Instanciamos la red entera
    neu_net = nn.NeuralNetwork()

    #Generar Red con pesos aleatorios
    neu_net.neuRandom(2,2,2) 
    
    for i in range(0,2):
        for j in range(0,2):
            vector= [i,j]
            #Evaluamos los vectores
            neu_net.evaluar(vector)
            #Imprimimos en consola los resultados
            neu_net.printG(vector)

    #Guardamos la red en formato JSON
    save(neu_net, 2, 'FeedFoward_Results/feed_forwards_random_results.json')
    print("Guardado en 'FeedFoward_Results/feed_forwards_random_results.json'") 
    neu_net = load(file)

    #Evaluamos la red e imprimimos los resultados de los vectores
    for i in range(0,2):
        for j in range(0,2):
            vector= [i,j]
            neu_net.evaluar(vector)
            neu_net.printG(vector)
    save(neu_net, 2, 'FeedFoward_Results/feed_forwards_results.json')
    print("Guardado en 'FeedFoward_Results/feed_forwards_results.json'")
