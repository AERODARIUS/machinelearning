from common import get_stats, load_data, select_data, select_data_casero, split_dataset_n, text_to_nums, normalize, evaluar_tasa_aciertos, names_to_nums
import numpy as np
import copy
import pandas as pd
import operator 
import matplotlib.pyplot as plt



dataset = load_data()
newDataset = dataset.replace(names_to_nums)
knn_dataset = newDataset


mapa = {}


def distancia(x,y,features):
    d = 0
    for i in range(features):
        if i == 10: #si es month
           d = d + min(pow((x[i] - y[i]),2),pow(12-(x[i] - y[i]),2))
        elif isinstance(x[i], str):
            d = d + 0 if x[i] == y[i] else d +1   
        else: #numero normales
            d = d + pow((x[i] - y[i]),2)
    return d


def getKNN(x,puntos,kList,features):
    valuesKNN = {}
    distancias = []
    for i in range(len(puntos)):
        d = distancia(x,puntos[i],features-1)
        distancias.append((puntos[i],d))
    distancias.sort(key=operator.itemgetter(1))
    for k in kList:
        aux = []
        for i in range(k):
            aux.append(distancias[i][0])
        valuesKNN[k] = copy.copy(aux)
    return valuesKNN

def classifyKNN(x,values,features):
    count0 = 0
    count1 = 0
    classifies = None
    for i in range(len(values)):
        if values[i][features] == 1:
            count1= count1+1
        else:
            count0 = count0+1
    if count0>count1:
        classifies = 0
    if count1>count0:
        classifies = 1
    return classifies

def KNN(x,puntos,kList,features):
    classifies = {}
    valuesKNN = getKNN(x,puntos,kList,features)
    for k in kList:
        classifies[k] = classifyKNN(x,valuesKNN[k],features)
    return classifies

def main():
    trainingList, test = split_dataset_n(knn_dataset,4,0.2)
    col_size = len(test.keys())-1
    testY = test.iloc[:, col_size:]

    promedio = {}
    validation = {}
    prediction = {}
    klist= [1,3,5,7,9]
    k_range = klist.copy()
    k_scores = []
    #para cada set del cross validation
    for setN in range(len(trainingList)):
        print("**********************")
        print("Validando con ", setN," entrenando con el resto")
        validation[setN] = {}
        val = trainingList[setN]
        valY = val.iloc[:, col_size]
        trainingCopy = copy.copy(trainingList)
        del trainingCopy[setN]
        training = pd.concat(trainingCopy)
        #para cada punto en validation le hago knn
        for index, row in val.iterrows():
            a = row.values.ravel()
            prediction[index] = (KNN(a,training.to_numpy(),klist,col_size))
        #a cada k le calculo los valores de score
        for k in klist:
            print('PROBANDO CON K = ', k)
            aux = []
            for index, row in val.iterrows():
                aux.append(prediction[index][k])
            accuracy, precision, recall, medidaF = get_stats(valY.values.ravel(),aux,1)
            print("Accuracy:",f'{round(accuracy*100, 2)}%')
            print("Precision:",f'{round(precision*100, 2)}%')
            print("Recall:",f'{round(recall*100, 2)}%')
            print("FMeasure:",f'{round(medidaF*100, 2)}%')
            validation[setN][k] = (accuracy,precision,recall,medidaF)
            # evaluar_tasa_aciertos(aciertos, 'el algoritmo de KNN casero')
            print("------------------------------------------")
    for k in klist:
        promedio[k] = [0,0,0,0]
        for setn in range(len(trainingList)):
            promedio[k][0] += validation[setn][k][0]
            promedio[k][1] += validation[setn][k][1]
            promedio[k][2] += validation[setn][k][2]
            promedio[k][3] += validation[setn][k][3]
        promedio[k] = list(map(lambda x: x / len(trainingList), promedio[k]))
        print(f'Promedio para k={k}: ', promedio[k][3])
        k_scores.append(promedio[k][3])
    # me quedo con el k que tenga mayor medidaF
    max = 0
    argMax = None
    for k in klist:
        if promedio[k][3] > max:
            max = promedio[k][3]
            argMax = k
    print("El k que tuvo mejor desempeño fue: " , argMax)
    newKList = [argMax]
    for index, row in test.iterrows():
        a = row.values.ravel()
        prediction[index] = (KNN(a,training.to_numpy(),newKList,col_size))
        #a cada k le calculo los valores de score
    for k in newKList:
        print('Evaluando desempeño con K = ', k)
        aux = []
        for index, row in test.iterrows():
            aux.append(prediction[index][k])
        accuracy, precision, recall, medidaF = get_stats(testY.values.ravel(),aux,1)
        print("Accuracy:",f'{round(accuracy*100, 2)}%')
        print("Precision:",f'{round(precision*100, 2)}%')
        print("Recall:",f'{round(recall*100, 2)}%')
        print("FMeasure:",f'{round(medidaF*100, 2)}%')
    
    plt.plot(k_range, k_scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Cross-validated Average F1 Score')
    plt.savefig('knn_parte4.png')


main()