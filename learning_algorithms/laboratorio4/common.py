import pandas as pd
import numpy as np
from math import log
import matplotlib.pyplot as plt

def load_data () :
  DATASET_FILE = './datasets_tareas_countries_of_the_world.csv'
  dataset = pd.read_csv(DATASET_FILE, sep=',')
  print('{} records read from {}\n{} attributes found'.format(dataset.shape[0], DATASET_FILE, dataset.shape[1]))
  print()
  return dataset

def text_to_nums(dataset) :
  dataset = dataset.replace({',':'.'},regex=True)
  for column in dataset:
      dataset[column] = pd.to_numeric(dataset[column])
  dataset = dataset.fillna(dataset.median())
  return dataset

def entropy (s) :
  if s.empty :
    return 0

  total = s.term_deposit.size
  svc = s.term_deposit.value_counts()

  if svc.size == 2 :  # hay 0s y 1s
    pos, neg = svc
    p_pos = pos / total
    p_neg = neg / total
    return -p_pos * log(p_pos, 2) - p_neg * log(p_neg,2)

  else : # svc == 1, o bien no hay 0s o bien no hay 1s
    return 0

def coseno_dist(x, y) :
  x_norm = np.linalg.norm(x)
  y_norm = np.linalg.norm(y)
  # 1 si el ángulo comprendido es cero
  # -1  si apuntan en dirección opuesta
  coseno_norm = np.sum(x*y) / (x_norm*y_norm) # -1 < coseno_norm < 1
  return 1 - coseno_norm

# Se asume que el dataset ya fue convertido a números
def minmax(dataset) :
  new_dataset = dataset

  for k in dataset.keys() :
    if not isinstance(dataset[k].iloc[0], str) :
      min = new_dataset[k].min()
      max = new_dataset[k].max() - min

      def normalize_cell(cell) :
        return (cell - min) / max

      new_dataset[k] = new_dataset[k].apply(normalize_cell)

  return new_dataset

colors = [
  'blue',
  'red',
  'green',
  'orange',
  'cyan',
  'saddlebrown',
  'violet',
  'darkgreen',
  'darkred',
  'palegreen',
  'darkkhaki',
  'dodgerblue',
  'indigo',
  'yellow',
  'pink',
  # 'black',
]

def isnan(array) :
  array_sum = np.sum(array)
  return np.isnan(array_sum)

# PCA obtenemos los datos
def PCA_plot(dataset, centroides, pca, uruguay = [], labels = np.array([])) :
  
  if labels.size > 0 :
    dataset2D = pca.transform(dataset).transpose()
    m = dataset2D.shape[1]
    _, ax = plt.subplots()
    ax.scatter(dataset2D[0,0:m], dataset2D[1,0:m])

    for i, txt in enumerate(labels):
      ax.annotate(txt, (dataset2D[0,0:m][i], dataset2D[1,0:m][i]))
  else :
    dataset2D = []
    centroides2D = pca.transform(centroides).transpose()
    ms = []
    cm = len(centroides)

    for d in dataset :
      d2D = pca.transform(d).transpose()
      ms.append(d2D.shape[1])
      dataset2D.append(d2D)

    for i, d2D in enumerate(dataset2D) :
      m = ms[i]
      plt.plot(d2D[0,0:m], d2D[1,0:m], 'o', markersize=3, color=colors[i], alpha=0.5)

    plt.plot(centroides2D[0, 0:cm], centroides2D[1, 0:cm], '.', markersize=8, color='black', alpha=1)

    if len(uruguay) > 0 :
      uruguay2D = pca.transform([uruguay])[0]
      plt.annotate("Uruguay", uruguay2D)

  plt.xlabel('nueva dimensión 1')
  plt.ylabel('nueva dimensión 2')
  plt.title('Instancias transformadas')

  plt.show()
