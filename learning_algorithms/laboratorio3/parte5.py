import pandas as pd
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB, ComplementNB, CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from common import load_data, select_data, text_to_nums, normalize, evaluar_tasa_aciertos, names_to_nums
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


dataset = load_data()

bayes_names = ['BernoulliNB', 'GaussianNB', 'MultinomialNB', 'ComplementNB']
bayes_model = {
  'BernoulliNB': BernoulliNB,
  'GaussianNB': GaussianNB,
  'MultinomialNB': MultinomialNB,
  'ComplementNB': ComplementNB
}

def bayes_scikit(model_name, train) :
  print(f'# Cross validate {model_name}')
  trainY = train.iloc[:, 16:]
  trainX = train.iloc[:, :16]

  bayes = bayes_model[model_name]()
  scores = cross_val_score(bayes, trainX, trainY.values.ravel(), cv=10, scoring='f1')

  print('scores mean:', scores.mean())

  print('\n-------------------------------------------\n')
  
  return scores.mean()

bayes_dataset = text_to_nums(dataset.replace(names_to_nums), True)
train, test = select_data(bayes_dataset, 0.2)
max_name = None
max_score = -1

for bayes_name in bayes_names :
  score = bayes_scikit(bayes_name, train)
  
  if score > max_score :
    max_score = score
    max_name = bayes_name

print(f'Mejor modelo elejido: {max_name} con un mean score {max_score}')

trainY = train.iloc[:, 16:]
trainX = train.iloc[:, :16]
testY = test.iloc[:, 16:]
testX = test.iloc[:, :16]
b_model = bayes_model[max_name]

bnb_model = b_model().fit(trainX, trainY.values.ravel())
predY = bnb_model.predict(testX)

print('Estadisticas para el algoritmo elegido como el mejor')
aciertos_scikit = accuracy_score(testY.values.ravel(), predY)
precision, recall, f1, support = precision_recall_fscore_support(testY.values.ravel(), predY, average='binary')
print("Accuracy:",f'{round(aciertos_scikit*100, 2)}%')
print("Precision:",f'{round(precision*100, 2)}%')
print("Recall:",f'{round(recall*100, 2)}%')
print("FMeasure:",f'{round(f1*100, 2)}%')
print()

print('\n===========================================\n')

print('# KNN')

knn_dataset = dataset.replace(names_to_nums)
knn_datasetY = knn_dataset.iloc[:, 16:]
knn_datasetX = pd.get_dummies(knn_dataset.iloc[:, :16])
# Al normalizar los datos la tasa de acierto del KNN de scikit dió peor
# knn_datasetX = normalize(knn_datasetX)
knn_dataset = knn_datasetX.join(knn_datasetY)
train, test = select_data(knn_dataset, 0.2)
col_size = len(train.keys())-1
trainY = train.iloc[:, col_size:]
trainX = train.iloc[:, :col_size]
testY = test.iloc[:, col_size:]
testX = test.iloc[:, :col_size]

k = 1
aciertos_scikit = 0
prev_aciertos_scikit = -1
max = 0
maxK = 0
k_range = []
k_scores = []

print(f'# KNeighborsClassifier: validando...')
while k < 10:
  knn = KNeighborsClassifier(n_neighbors=k)
  scores = cross_val_score(knn, trainX, trainY.values.ravel(), cv=4, scoring='f1')
  mean = scores.mean()
  if mean > max:
    max = mean
    maxK = k
  k_range.append(k)
  k_scores.append(mean)
  k += 2

print("Se creó un archivo knn_parte5")
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-validated Average F1 Score')
plt.savefig('knn_parte5.png')

print(f'El mejor K calculado usando validacion fue {maxK}')
print('Se procede a realizar la prueba con el conjunto Test')


KNN = KNeighborsClassifier(n_neighbors=maxK)
knn_model = KNN.fit(trainX, trainY.values.ravel())
predY = knn_model.predict(testX)

aciertos_scikit = accuracy_score(testY.values.ravel(), predY)
precision, recall, f1, support = precision_recall_fscore_support(testY.values.ravel(), predY, average='binary')
print("Accuracy:",f'{round(aciertos_scikit*100, 2)}%')
print("Precision:",f'{round(precision*100, 2)}%')
print("Recall:",f'{round(recall*100, 2)}%')
print("FMeasure:",f'{round(f1*100, 2)}%')
print()

