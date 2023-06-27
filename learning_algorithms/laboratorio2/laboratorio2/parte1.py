import pandas as pd
from scipy.sparse import data
from sklearn import model_selection
from math import log
from tree_class import Tree
import random
import matplotlib.pyplot as plt
import sys

cm = 1/2.54

random.seed() # Random pero predecible

# Calcula la entropía de un dataset `s`
def entropy (s) :
  if s.empty :
    return 0

  total = s.target.size
  svc = s.target.value_counts()

  if svc.size == 2 :  # hay 0s y 1s
    pos, neg = svc
    p_pos = pos / total
    p_neg = neg / total
    return -p_pos * log(p_pos, 2) - p_neg * log(p_neg,2)

  else : # svc == 1, o bien no hay 0s o bien no hay 1s
    return 0

# Realiza la particióin utilizando el atributo/columna del dataset
# Retorna el valor utilizado para la partición
def split_dataset (dataset, column) :
  s_size = dataset.size
  col_data = dataset[column]
  val_count = col_data.value_counts().sort_index().items()
  last_left = -1
  g_max = -1

  # Probar todas las posibles particiones sobre ese atributo
  # y quedarse con la de mayor ganancia
  for value, _ in val_count :
    p_left = dataset[dataset[column] <= value]
    e_left = entropy(p_left)
    pl_size = p_left.size
    p_right = dataset[dataset[column] > value]
    e_right = entropy(p_right)
    pr_size = p_right.size
    # No calclamos la entropía de dataset, porque no es necesario para comparar
    g = - (pl_size/s_size) * e_left - (pr_size/s_size) * e_right
    if g > g_max :
      g_max = g
      last_left = value

  p_left = dataset[dataset[column] <= last_left]
  p_right = dataset[dataset[column] > last_left]

  if p_right.empty :
    return g_max, last_left, p_left, p_right

  first_right = p_right[column].value_counts().sort_index().first_valid_index()
  # Punto medio entre el último elemento de `partition`
  # y el primer elemento de `rest`
  mid = (last_left + first_right) / 2

  # print('@@@@@@@@@@')
  # print(g_max)
  # print(mid)
  # print(p_left)
  # print(p_right)
  return g_max, mid, p_left, p_right



def get_child_nodes (dataset) :
  attributes = dataset.keys()
  d_size = dataset.target.size
  d_entropy = entropy(dataset)
  t = Tree()

  # print('samples = ', d_size)
  # print('entropy = ', d_entropy)

  # Si no tengo más atributos, decidir si pasó o no en base a la columna target
  if attributes.size == 1 or d_size == 0 or d_entropy == 0 :
    counts = dataset.target.value_counts().sort_index()

    t.data = 'leaf'

    if counts.size == 0 :
      t.treshold = round(random.random())
    elif counts.size == 1 :
      t.treshold = counts.keys()[0]
    else :
      if counts[1] == counts[0] :
        t.treshold = round(random.random())
      else :
        t.treshold = 1 if counts[1] > counts[0] else 0
    # print('================================================================')
    return t


  # Calcular la ganancia de hacer una partición en cada columna
  g_max = -100  # ganacia de la partición
  c_val_max = 0 # valor del atributo para la partición
  col_max = ''  # atributo con el que se parte
  p1_max = None # nodo izquierdo
  p2_max = None # nodod derecho

  # Obtener la mejor particion de cada atributo,
  # luego quedarse con la mejor de todas esas
  for key in attributes.drop('target') :
    # Fijando el atributo, obtener la mejor partición
    g, cond_val, p1, p2 = split_dataset(dataset, key)
    if g > g_max :
      g_max = g
      c_val_max = cond_val
      col_max = key
      p1_max = p1
      p2_max = p2

  # print('Condición elegida:', f'{col_max} <= {c_val_max}', 'con ganancia', g_max)
  
  # print('================================================================')

  # Genero la partición, quitando la columna con la que se hizo la partición
  # para evitar volver a partir con la misma columna
  left_child = p1_max[p1_max.keys().drop(col_max)]
  right_child = p2_max[p2_max.keys().drop(col_max)]

  t.left = get_child_nodes(left_child)
  t.right = get_child_nodes(right_child)
  t.data = col_max # f'({col_max} <= {c_val_max}, {d_entropy}, {d_size})', 
  t.treshold = c_val_max

  # Retornar árbol con las dos ramas
  return t

def predict(case, tree) :
  tree_iter = tree

  while tree_iter.data != 'leaf' :
    col_name = tree_iter.data
    treshold = tree_iter.treshold
    value = case[col_name]
    tree_iter = tree_iter.left if value <= treshold else tree_iter.right
  
  return tree_iter.treshold

def predict_all(dataset, tree) :
  size = dataset.target.size
  predictions = []

  for  i in range(size) :
    prediction = predict(dataset.iloc[i],tree)
    predictions.append(prediction)

  return predictions

def accuracy (current, expected) :
  size = len(current)
  matches = []
  acc = 0

  for  i in range(size) :
    m = (current[i] == expected.iat[i])
    matches.append(m)

    if m :
      acc += 1

  return matches, ((acc / size) * 100)

def load_data () :
  # Cargamos los datos
  DATASET_FILE = './material_adicional_heart_disease_uci.csv'
  dataset = pd.read_csv(DATASET_FILE, sep=',')
  print("{} records read from {}\n{} attributes found".format(dataset.shape[0], DATASET_FILE, dataset.shape[1]))
  return dataset

def select_data (dataset, ts) :
  # Elegimos un 80% de las columnas al azar
  # El conjunto al azar siempre es el mismo, para que sea uno diferente quitar `random_state=42`
  return model_selection.train_test_split(dataset, test_size=ts, random_state=42)

# Puede ser que algún caso borde esté fallando
# Por ejemplo cuanod tengamos solo casos positivos o negativos, o pocas muestras
# Esto lo vamos a ver cuando lleguemos a los nodos finales
def main() :
  dataset = load_data()
  train, test = select_data(dataset, 0.2)

  # print('RAÍZ')
  tree = get_child_nodes(train)
  # tree.display()

  predictions = predict_all(test, tree)
  # print(predictions)
  m, a = accuracy(predictions, test.target)
  # print(m)
  print(a)
  tree.display()

main()

def multi_test () :
  dataset = load_data()
  training_size = []
  acc_list = []
  tree_sizes = []
  for x in  range(1,11) :
    i = (x*10)-1
    test_size = (100 - i)/100
    training_size.append(i)
    train, test = select_data(dataset, test_size)
    tree = get_child_nodes(train)
    predictions = predict_all(test, tree)
    m, a = accuracy(predictions, test.target)
    tree_size = tree.getNodeCount()
    tree_sizes.append(tree_size)
    acc_list.append(a)
  
  tree_sizes.sort()
  plt.subplots(figsize=(30*cm, 10*cm))
  plt.plot(training_size, acc_list, '-')
  plt.xlabel('Porcentaje de entrenamiento')
  plt.ylabel('Aciertos')
  plt.show()

multi_test()