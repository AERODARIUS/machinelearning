import pandas as pd
import numpy as np
from sklearn import model_selection
from math import log
import argparse
import numpy as np


def load_data () :
  DATASET_FILE = './dataset.csv'
  dataset = pd.read_csv(DATASET_FILE, sep=',')
  print('{} records read from {}\n{} attributes found'.format(dataset.shape[0], DATASET_FILE, dataset.shape[1]))
  print()
  return dataset

def select_data (dataset, ts) :
  # Elegimos según ts, un porcentaje de las columnas de forma aleatoria
  # El conjunto al azar siempre es el mismo, para que sea uno diferente quitar `random_state=42`
  train, test = model_selection.train_test_split(dataset, test_size=ts, random_state=42, stratify=dataset['term_deposit'])

  print('{} muestras para entrenamiento, {} muestras para pruebas'.format(train.shape[0], test.shape[0]))
  print()

  return train, test

def select_data_casero (dataset, ts):
  df = dataset
  #Calculos
  count_row = dataset.shape[0]
  # ts puede ser un porcentaje o la cantidad de filas
  testNumber = ts if ts > 1 else (count_row * ts)
  N=testNumber
  #Estratificacion del conjunto test
  test = dataset.groupby('term_deposit', group_keys=False).apply(lambda x: x.sample(int(np.rint(N*len(x)/len(df))))).sample(frac=1)
  #El resto es para train
  train = dataset.merge(test, indicator=True, how='outer', left_index=True, right_index=True, suffixes=(None,"MERGECOLUMN")).query('_merge=="left_only"').drop('_merge', axis=1)
  train = train[train.columns.drop(list(train.filter(regex='MERGECOLUMN')))]
  return train, test

def parse_args (desc) :
  parser = argparse.ArgumentParser(description=desc)

  parser.add_argument('--verbose', '-v', dest='v', action='store_true', help='Imprime en la consola el avance del entrenamiento y pruebas, admás muestra los gráficos.')

  args = parser.parse_args()

  return args.v

def evaluar_tasa_aciertos (a, name) :
  if a > 79 :
    print(f'Consideramos que {name} tuvo una buena tasa de aciertos')
  elif a > 50 :
    print(f'Consideramos que para {name} la tasa de aciertos no fue tan buena')
  else :
    print(f'La tasa de aciertos fue mala para {name}')

def text_to_nums(dataset, skip_last = False) :
  mapping = {}
  keys = dataset.keys()

  if skip_last :
    keys = keys.drop('term_deposit')

  for k in keys :
    is_num = str(dataset[k].dtype) in ('int64', 'float64')
    if not is_num :
      keys = dataset[k].unique()
      mapping[k] = dict(zip(keys, range(len(keys))))

  return dataset.replace(mapping)

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

# Se asume que el dataset ya fue convertido a números
def normalize(dataset) :
  new_dataset = dataset

  for k in dataset.keys().drop('term_deposit'):
    if not isinstance(dataset[k].iloc[0], str):
      min = new_dataset[k].min()
      max = new_dataset[k].max() - min

      def normalize_cell(cell) :
        return (cell - min) / max

      new_dataset[k] = new_dataset[k].apply(normalize_cell)

  return new_dataset


names_to_nums = {
  # already numbers: sibilings, campaign, pdays, previous
  'age_range' : {
    'Young':  0,
    'Middle_Age': 1,
    'Old_Adult': 2,
  },
  'balance_range': {
    'low':  0,
    'medium': 1,
    'high': 2,
  },
  'education': {
    'tertiary': 3,
    'secondary': 2,
    'primary': 1,
    'unknown': 0 # podríamos asignarle el valro más probable que es el secondary
  },
  'default': {
    'yes': 1,
    'no': 0
  },
  'housing': {
    'yes': 1,
    'no': 0
  },
  'loan': {
    'yes': 1,
    'no': 0
  },
  'month': {
    'may': 5,
    'jul': 7,
    'aug': 8,
    'jun': 6,
    'nov': 11,
    'apr': 4,
    'feb': 2,
    'jan': 1,
    'oct': 10,
    'sep': 9,
    'mar': 3,
    'dec': 12
  },
  'duration': {
    'short': 0,
    'medium':  1,
    'long': 2,
  },
  'term_deposit': {
    'yes': 1,
    'no': 0
  }
}





def sprlit_n(tv_datasets, n) :
  ts_n = round(tv_datasets.shape[0] / n)
  tvd_list = []

  for _ in range(n-1) :
    tv_datasets, tv_dataset = select_data_casero(tv_datasets, ts_n)
    tvd_list.append(tv_dataset)

  tvd_list.append(tv_datasets)

  return tvd_list

# particiona el dataset en n+1 conjuntos
# return ([train_validate_1, ..., train_validate_n], test)
# train_validate_i se usan para validar y testear, todos del mismo tamaño
def split_dataset_n(dataset, n, ts) :
  tv_datasets, test = select_data_casero(dataset, ts)
  tvd_list = sprlit_n(tv_datasets, n)
  return tvd_list, test

# beta es un hiperparámetro
# cuanta más importancia se le da al recall respecto a la precisión
# se pasa como parámetro para usar uno diferente en knn y bayes
# ver https://en.wikipedia.org/wiki/F-score

# expected y predicted son arreglos de 'yes' y 'no'

# Recall: The ability of a model to find all the relevant cases within a data set.
# Mathematically, we define recall as the number of true positives divided by the
# number of true positives plus the number of false negatives.

# Precision: The ability of a classification model to identify only the relevant
# data points. Mathematically, precision the number of true positives divided by
# the number of true positives plus the number of false positives.

def get_stats(expected, predicted, beta) :
  beta2 = beta * beta
  total = len(expected)
  Vp = 0 # Verdadero positivo
  Fp = 0 # Falso positivo
  Vn = 0 # Verdadero negativo
  Fn = 0 # Falso negativo


  for i in range(total) :
    e = expected[i]
    p = predicted[i]

    if p == 'yes' or p == 1 :
      if e == p :
        Vp += 1
      else :
        Fp += 1
    else :
      if e == p :
        Vn += 1
      else :
        Fn += 1

  accuracy = (Vp + Vn) / total
  precision = -1 if (Vp + Fp == 0) else (Vp / (Vp + Fp))
  recall = -1 if (Vp + Fn == 0) else (Vp / (Vp + Fn))
  medidaF = -1
  
  if precision != -1 and recall != -1 and (precision + recall) != 0 :
    medidaF = ((1 + beta2) * precision * recall) / (beta2 * precision + recall)

  return accuracy, precision, recall, medidaF


def select_1_1_ratio(dataset, td, class_value = 1) :
  positive_ds = dataset[dataset.term_deposit == class_value]
  negative_ds = dataset[dataset.term_deposit != class_value]
  train_yes, test_yes = select_data_casero(positive_ds, td)
  test_no_size = negative_ds.term_deposit.size - train_yes.term_deposit.size
  train_no, test_no = select_data_casero(negative_ds, test_no_size)

  return pd.concat([train_yes, train_no]), pd.concat([test_yes, test_no])
