from Classes import NetLayer as ntl
from Classes import NeuralNetwork as nn
import t4_back_propagation as gp
import matplotlib.pyplot as plt
from tabulate import tabulate
from random import uniform
import pandas as pd
import numpy as np
import json
import sys


def neuralArchitecture(dataN, dataV,cNeuronH, cInH, cNeuronO):
    roundWithoutDecrement, epsilon = 3, 0.5
    net = nn.NeuralNetwork() 

    #(cantidad de neuronas, cantidad de entradas)
    net.AddHiddenLayer(cNeuronH,cInH)

    #(cantidad de neuronas)
    net.AddOutLayer(cNeuronO) 
    #print(dataN)
    for i in range(len(dataN)):
        vector = data[i]
        outs = dataV[i]
        mseMax, mseMin, mseProm, nCol = [],[],[],[]
        cont = 0
        cont += 1
        nCol.append(cNeuronH)
        errorA, MSE = 1, []
        
        for j in range(50):
            net.evaluar(vector)
            net = gp.gradient(net,vector)
            net = gp.weightUpdate(net,vector)

            #Calculamos el MSE actual
            ultimasN = net.layers[len(net.layers)-1].neurons
            error = (np.square(outs[i] - [ultimasN[0][2],ultimasN[1][2]])).mean()
            #Vemos si el error ha aumentado, detener el proceso
            if error > errorA: 
                #print(f"El error empezó a aumentar. Se detiene el proceso.\nLa época: {j+1} no se guardará")
                break
           #Si los errores actual con el pasado es igual:
            if(error == errorA):
                #Si llego al limite especificado, termine
                if(roundSuccesful == roundWithoutDecrement):
                    #print("Llegó al limite de rondas sin incrementos establecidos")
                    break
                #si no, continue, buen joven
                else:
                    roundSuccesful+=1
            else:
                if((errorA - error) <= epsilon): break
                #Significa que el error es menor, así que pierde la racha de errores iguales
                roundSuccesful = 0
            
            #Actualizamos el errors anterior con el actual
            errorA = error
            
            #Añadimos el error al arreglo de MSE
            MSE.append(errorA)
            
            #Imprimimos el MSE se especifico como requerimieto
            print(tabulate([[j,error]], headers=['ÉPOCA','MSE'],tablefmt='fancy_grid', showindex="never", colalign=("center",)),"\n")
    
        #Con np calculamos el max, min y promedio de un solo
        mseMax.append(np.amax(MSE))
        mseMin.append(np.amin(MSE))
        mseProm.append(np.mean(MSE))
        
        #guardamos los resultados de la red
        save(net,2,'Clasicacion_Result/JSON/results_net_'+str(cNeuronH)+'.json')
        
        #Se imprime en consola
        print(tabulate([[cont,np.amax(MSE),np.amin(MSE),np.mean(MSE)]], headers=['Net','MSE_MAX','MSE_MIN','MSE_PROM'],tablefmt='fancy_grid', showindex="never", colalign=("center",)),"\n")


                        
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


def saveGraphic(mseMax,mseProm,mseMin,vector):
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
    plt.savefig("Clasicacion_Result/Graphics/Results_"+str(vector[0])+str(vector[1])+".png")
    #Limiamos el plot para el próximo gráfico
    plt.clf()

    
if __name__ == "__main__":
    file = sys.argv[1] if len(sys.argv) == 1 else './Datos/part3_data_train.csv'
    file2 = sys.argv[2] if len(sys.argv) == 1 else './Datos/part3_data_test.csv'

    trainingData = pd.read_csv(file, engine='python')
    testData = pd.read_csv(file2, engine='python')

    hotEncodingT = trainingData.replace({
        'arroz_frito':'1,0,0,0,0','ensalada':'0,1,0,0,0','pollo_horneado':'0,0,1,0,0', 'hamburguesa':'0,0,0,1,0','pizza':'0,0,0,0,1'
        })
    hotEncodingV = testData.replace({
        'arroz_frito':'1,0,0,0,0','ensalada':'0,1,0,0,0','pollo_horneado':'0,0,1,0,0', 'hamburguesa':'0,0,0,1,0','pizza':'0,0,0,0,1'
        })
    

    #Segmentamos
    data = hotEncodingT.iloc[:,0:5].values
    dataV = hotEncodingV.iloc[:,0:5].values
    
    #Normamizamos los datos
    dataN=(data-data.min())/(data.max()-data.min())
    dataVN=(dataV-dataV.min())/(dataV.max()-dataV.min())

    #5 entradas, una capa oculta con 4 neuronas, y una capa de salida con 5 neuronas.
    net1 = neuralArchitecture(dataN,dataVN,4,5,5)

    #5 entradas, una capa oculta con 16neuronas,y una capa de salida con 5 neuronas.
    net2 = neuralArchitecture(dataN,dataVN,16,5,5)

    #5 entradas, una capa oculta con 32neuronas,y una capa de salida con 5 neuronas
    net3 = neuralArchitecture(dataN,dataVN,32,5,5)

    #5 entradas, una capa oculta con 64neuronas,y una capa de salida con 5 neuronas.
    net4 = neuralArchitecture(dataN,dataVN,64,5,5)