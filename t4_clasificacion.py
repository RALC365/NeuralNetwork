from Classes import NetLayer as ntl
from Classes import NeuralNetwork as nn
import t4_back_propagation as gp

import sys
import numpy as np
from random import uniform
import json
import pandas as pd
import matplotlib.pyplot as plt


    #py Parte3_4.py part3_data_train.csv
if __name__ == "__main__":
    #archivos JSON - parte1
    file = ''
    if len(sys.argv) == 1:
        file = './Datos/part3_data_train.csv'
    else:
        file = sys.argv[1]
    
    trainingData = pd.read_csv(file, engine='python')

    cambio = trainingData.replace({
        'arroz_frito':'1,0,0,0,0',
        'ensalada':'0,1,0,0,0',
        'pollo_horneado':'0,0,1,0,0',
        'hamburguesa':'0,0,0,1,0',
        'pizza':'0,0,0,0,1'
        })

    data = cambio.iloc[:,0:5].values
    datosNormalizados=(data- data.min())/(data.max()- data.min())

    #crear la red - se inicializa con los pesos entre -1 y 1
    net = nn.NeuralNetwork()  #red = RedCompleta()
    net.AddHiddenLayer(4,5)  #red.AddCapaOculta(4,5) #4 neuronas, 5 entradas
    net.AddOutLayer(5)  #red.AddCapaSalida(5) #5 neuronas

    #por lo momentos tarda 5 minutos con una red
    for i in datosNormalizados[:,:]:
        for j in range(50): #rondas
            net.evaluar(i) #red.evaluar(i)
            gp.calculoGradiente(net,i)    #calculoGradiente(red,i)
            gp.cambiarPesos(net,i)    #cambiarPesos(red,i)

    net2 = nn.NeuralNetwork()    #red2 = RedCompleta()
    net2.AddHiddenLayer(16,5)   #red2.AddCapaOculta(16,5) #16 neuronas, 5 entradas
    net2.AddOutLayer(5)    #red2.AddCapaSalida(5) #5 neuronas

    net3 = nn.NeuralNetwork()
    net3.AddHiddenLayer(32,5)
    net3.AddOutLayer(5)

    net4 = nn.NeuralNetwork()
    net4.AddHiddenLayer(64,5)
    net4.AddOutLayer(5)