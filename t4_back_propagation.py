from Classes import NetLayer as ntl
from Classes import NeuralNetwork as nn
from tabulate import tabulate
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import json

# Ignora las advertencias
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

                        
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

def BackPropagation(datat, cEpochMax, roundWithoutDecrement, epsilon, validationfile, dataValidation):
    data, roundSuccesful, nets = {}, 0, []
    #Separamos de un solo la training data
    ins = datat.iloc[:,0:2].values
    outs = datat.iloc[:,2:4].values
    
    for i in range(20):
        net = nn.NeuralNetwork()
        
        #Inicializamoos los valores solo al inicio
        net.neuRandom(2,2,2)
        
        #Añadimos la red al arreglo
        nets.append(net)
    
    
    #evaluamos cada red por los 4 vectores de entrada
    print("Working...")
    for i in range(len(ins)):
        vector = ins[i]
        mseMax, mseMin, mseProm, nCol = [],[],[],[]
        cont = 0
        print(f"--------------------{i}-------------------------")
        for net in nets:
            cont += 1
            nCol.append(nets.index(net)+1)
            errorA, MSE = 1, []
            #rondas, por defecto son 100, a menos que el usuari mande un valor
            for j in range(cEpochMax): 
                net.evaluar(vector)
                gradient(net,vector)
                weightUpdate(net,vector)
                
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
            save(net,2,'BackPropagation_Results/JSON/results_net_'+str(nets.index(net)+1)+'.json')
            
            #Se imprime en consola
            print(tabulate([[cont,np.amax(MSE),np.amin(MSE),np.mean(MSE)]], headers=['Net','MSE_MAX','MSE_MIN','MSE_PROM'],tablefmt='fancy_grid', showindex="never", colalign=("center",)),"\n")

        #Formateamos
        data = {'NET': nCol, 'MSE_MIN': mseMin, 'MSE_MAX': mseMax, 'MSE_PROM': mseProm}
        archivo = pd.DataFrame(data, columns=['NET','MSE_MIN','MSE_MAX','MSE_PROM'])
        
        #Se imprime en consola
        print(tabulate(archivo, headers=['NET','MSE_MAX','MSE_MIN','MSE_PROM'],tablefmt='fancy_grid', showindex="always", colalign=("center",)))
        
        #Guardamos
        archivo.to_csv('BackPropagation_Results/CSV/inputs_net_'+str(vector[0])+str(vector[1])+'.csv', index=False)
        saveGraphic(mseMax,mseProm,mseMin,vector)

    print("Los resultados se guardaron en 'BackPropagation_Results/'")

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
    plt.savefig("BackPropagation_Results/Graphics/Results_"+str(vector[0])+str(vector[1])+".png")
    #Limiamos el plot para el próximo gráfico
    plt.clf()

def weightUpdate(net, vector):
    a0, alfa = 1, 0.1     
    for i in range(len(net.layers)):
        layer = net.layers[i]
        for neuron in layer.neurons:
            for j in range(len(neuron[1])):
                if j==0:
                    neuron[1][j]+=neuron[3]*alfa*a0
                if i >0 and j>0:
                    nBefore = net.layers[i-1].neurons
                    neuron[1][j]+=neuron[3]*alfa*nBefore[j-1][3]
                else:
                    if j>0:
                        vVector = vector[j-1]
                        neuron[1][j]+=neuron[3]*alfa*vVector
    return net

def gradient(net,vector):
    for i in range (len(net.layers)):
        ind = (len(net.layers)-1)-i
        layer =net.layers[ind]
        for j in range(len(layer.neurons)):
            a = 0
            neuron = layer.neurons[j]
            g = neuron[2]*(1-neuron[2])
            if ind ==(len(net.layers)-1):
                a = vector[j]-neuron[2]
            else:
                before = net.layers[ind+1].neurons
                for nBefore in before:
                    a = a + (nBefore[1][0]*nBefore[3])
            neuron[3] = g*a
    return net

if __name__ == "__main__":
    #Validación de argumentos de entrada. Se quizo hacer one-line pero no
    #Datos de entrenamiento - mandatorio
    lure =['./Datos/part2_train_data.csv', 100, -1, -1, '']

    for i in range(1,len(sys.argv)):
        lure[i-1] = sys.argv[i]

    #file = './Datos/part2_train_data.csv' if len(sys.argv) == 1 else sys.argv[1]
    file = lure[0]
    
    #Número máximo de épocas - mandatorio
    cEpochMax = int(lure[1])
    
    #Cantidad de rondas sin decremento - opcional
    roundWithoutDecrement = lure[2]
    
    #Valor epsilon - opcional
    epsilon = float(lure[3])

    #Datos de validación - opcional
    validationfile = lure[4]

    #leemos la red del archivo
    data = pd.read_csv(file, engine='python')
    dataValidation = None if validationfile == '' else pd.read_csv(validationfile, engine='python')

    #Red Neuronal
    BackPropagation(data, cEpochMax, roundWithoutDecrement, epsilon, validationfile, dataValidation)
    
    
