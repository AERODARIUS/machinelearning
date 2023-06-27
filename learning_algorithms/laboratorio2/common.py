
import pandas as pd
from sklearn import model_selection
import argparse

def load_data (is_verbose = False) :
  # Cargamos los datos
  DATASET_FILE = './material_adicional_heart_disease_uci.csv'
  dataset = pd.read_csv(DATASET_FILE, sep=',')
  print("{} registros cargados de {}\n{} atributos encontrados".format(dataset.shape[0], DATASET_FILE, dataset.shape[1]))
  return dataset

def select_data (dataset, ts, is_verbose = False) :
  # Elegimos según ts, un porcentaje de las columnas de forma aleatoria
  # El conjunto al azar siempre es el mismo, para que sea uno diferente quitar `random_state=42`
  train, test = model_selection.train_test_split(dataset, test_size=ts, random_state=42)
  print("{} muestras para entrenamiento, {} muestras para pruebas".format(train.shape[0], test.shape[0]))

  return train, test

def parse_args (desc) :
  parser = argparse.ArgumentParser(description=desc)

  parser.add_argument('--verbose', '-v', dest='v', action='store_true', help='Imprime en la consola el avance del entrenamiento y pruebas, admás muestra los gráficos.')
  parser.add_argument('--multi', '-m', dest='m', action='store_true', help='Por defecto se realiza un único experimento entrenando con un 80%% de los datos y realizando un test con el 20%% restante. Al utilizar esta opción se realizan varias preubas con varias combinaciones.')
  parser.add_argument('--tree', '-t', dest='t', action='store_true', help='Imprime el árbol generado en la consola.')
 
  args = parser.parse_args()

  return args.v, args.m, args.t

def evaluar_tasa_aciertos (a, name) :
  if a > 79 :
    print(f'Consideramos que {name} tuvo una buena tasa de aciertos')
  elif a > 50 :
    print(f'Consideramos que para {name} la tasa de aciertos no fue tan buena')
  else :
    print(f'La tasa de aciertos fue mala para {name}')
    

