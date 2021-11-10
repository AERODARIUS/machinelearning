from math import log
from tree_class import Tree
from common import parse_args
import random

random.seed()

is_verbose, is_multi, show_tree = parse_args('Laboratorio 2')

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
  g_max = -100

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

  return g_max, mid, p_left, p_right

def get_child_nodes (dataset) :
  attributes = dataset.keys()
  d_size = dataset.target.size
  d_entropy = entropy(dataset)
  t = Tree()

  if is_verbose :
    print('muestras = ', d_size)
    print('entropía = ', d_entropy)

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
    
    if is_verbose :
      print('================================================================')

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

  if is_verbose :
    print('Condición elegida:', f'{col_max} <= {c_val_max}')
    print('================================================================')

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
