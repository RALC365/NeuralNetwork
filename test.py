from Classes import NetLayer as ntl
from Classes import NeuralNetwork as nn

import sys
import numpy as np
from random import uniform
import json
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
#Neural Network classes



#Metodoc cambiado
#metodo para cargar datos los datos del archivo JSON
def cargar(file):
    redJson = nn.NeuralNetwork()
    with open(file) as archivo:
        entrada = json.load(archivo)
        entradas = entrada['entradas']
        print("entradas: ",entradas)

        #redJson = red = RedCompleta()
        for capa in entrada['capas']:
            capaNueva = ntl.NetLayer()
            for neurona in capa['neuronas']:
                w = neurona['pesos']
                capaNueva.neurons = np.append(capaNueva.neurons,[1,w,0,0])
            redJson.layers = np.append(redJson.layers, capaNueva)

    for i in range(0,2):
        for j in range(0,2):
            vector= [i,j]
            redJson.evaluar(vector)
            redJson.imprimirSalidas(vector)


#metodo cambiado
#metodo para guardar los datos en el archivo JSON
def guardar(red, X, ruta):
    salida ={}
    salida['entradas'] = X
    salida['capas'] = []
    for i in range(len(red.layers)):
        neuronaJ = {}
        neuronaJ['neuronas'] = []
        for j in range(len(red.layers[i].neurons)):
            neurona = red.layers[i].neurons[j]
            pesos = {}
            pesos['pesos'] = neurona[1]
            neuronaJ['neuronas'].append(pesos)
        salida['capas'].append(neuronaJ) 
    with open(ruta, 'w') as file:
        json.dump(salida, file)



#solo inicia si es el proceso inicial#
if __name__ == "__main__":
    #python t4_feed_forward.py part1.red.prueba.json part2_train_data.csv

    file = ''
    if len(sys.argv) == 1:
        file = './Datos/part1_red_prueba.json'
    else:
        file = sys.argv[1]

    #archivos JSON - parte1
    archivo1 = sys.argv[1]
    
    neu_net = nn.NeuralNetwork()
    #tiene dos entradas, dos capas y dos neuronas x capas
    
    neu_net.generar(2,2,2) 
    #vectores de entrada
    print("-----------------> PARTE 1 <---------------------")
    for i in range(0,2):
        for j in range(0,2):
            vector= [i,j]
            neu_net.evaluar(vector)
            neu_net.imprimir_salidas(vector)

    guardar(neu_net, 2, 'ArchivosJSON/salida_part1.json')
    print("El archivo se guardo exitosamente! (revisar ArchivosJSON/salida_part1.json)") 
    #cargar(archivo1)

    #print("")
    #print("-----------------> PARTE 2 <---------------------")
    #BackPropagation(trainingData)

    
