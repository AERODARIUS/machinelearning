import pandas as pd
from sklearn import utils, model_selection, tree, metrics
import matplotlib.pyplot as plt

cm = 1/2.54

# Cargamos los datos
DATASET_FILE = './material_adicional_heart_disease_uci.csv'
dataset = pd.read_csv(DATASET_FILE, sep=',')
print("{} records read from {}\n{} attributes found".format(dataset.shape[0], DATASET_FILE, dataset.shape[1]))
print(dataset.age.value_counts())

# Elegimos un 80% de las columnas al azar
# El conjunto al azar siempre es el mismo, para que sea uno diferente quitar `random_state=42`
train, test = model_selection.train_test_split(dataset, test_size=0.6, random_state=42)

print("{} samples for training, {} samples for testing".format(train.shape[0], test.shape[0]))

## Parte B ##

# Crear el arbol de desición
input_cols = dataset.columns[0:13]
my_tree = tree.DecisionTreeClassifier(criterion="entropy")
my_tree = my_tree.fit(train[input_cols], train.target)

##  Evaluamos los ejemplos de testeo, y los comparamos con los reales para calcular el acierto, precisión, etc.

# predecimos los ejemplos del conjunto de test

test_pred = my_tree.predict(test[input_cols])

# y los comparamos contra los "reales"
print("\nAcierto:", metrics.accuracy_score(test.target, test_pred))

# veamos precisión, recuperación...
print(metrics.classification_report(test.target, test_pred))

# # Imprimir el árbol de desición
# fig = plt.figure(figsize=(100,100))
# tree.plot_tree(my_tree, filled=True, fontsize=12)
# plt.show()
# fig.savefig('./parte2.out.png', dpi=100)

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

    input_cols = dataset.columns[0:13]
    my_tree = tree.DecisionTreeClassifier(criterion="entropy")
    my_tree = my_tree.fit(train[input_cols], train.target)
    test_pred = my_tree.predict(test[input_cols])
    tree_size = my_tree.tree_.node_count
    tree_sizes.append(tree_size)
    a = metrics.accuracy_score(test.target, test_pred)
    acc_list.append(a*100)
  tree_sizes.sort()
  print(tree_sizes)
  plt.subplots(figsize=(30*cm, 10*cm))
  plt.plot(training_size, acc_list, '-')
  plt.xlabel('Porcentaje de entrenamiento')
  plt.ylabel('Aciertos')
  plt.show()

multi_test()
