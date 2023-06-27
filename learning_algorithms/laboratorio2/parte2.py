from sklearn import tree as sk_tree, metrics
import matplotlib.pyplot as plt
from common import parse_args, load_data, select_data, evaluar_tasa_aciertos
from id3_nuestro import get_child_nodes, predict_all, accuracy

is_verbose, is_multi, show_tree = parse_args('Parte 2: Algoritmo de sklearn')

def compare (aciertos_nuestro, aciertos_scikit) :
  if abs(aciertos_nuestro - aciertos_scikit) < 0.001 :
    print('Nuestro algoritmo tuvo una tasa de acierto similar a la de scikit, por lo que se puede decir que ambos tuvieron un desempeño similar')
  elif aciertos_nuestro > aciertos_scikit :
    print('Nuestro algoritmo tuvo una tasa de acierto mayor a la de scikit por lo que se puede decir que tuvo un mejor desempeño que scikit')
  else :
    print('Nuestro algoritmo tuvo una tasa de acierto menor a la de scikit por lo que se puede decir que tuvo un desempeño peor que el de scikit')


## Parte B ##
def main () :
  # Cargamos los datos
  dataset = load_data(is_verbose)

  # Elegimos un 80% de las columnas al azar
  train, test = select_data(dataset, 0.2)

  # Nuestro algoritmo
  tree = get_child_nodes(train)
  predictions = predict_all(test, tree)
  m, aciertos_nuestro = accuracy(predictions, test.target)

  if is_verbose :
    print()
    print('Predicciones')
    print('-------------')
    for p in m :
      print('ACIERTO' if p else 'FALLO')

  print()
  print('Precisión de aciertos de nuestro algoritmo:', f'{round(aciertos_nuestro, 2)}%')
  evaluar_tasa_aciertos(aciertos_nuestro, 'nuestra implementación del algoritmo ID3')

  print()

  # Crear el arbol de desición
  input_cols = dataset.columns[0:13]
  my_tree = sk_tree.DecisionTreeClassifier(criterion="entropy")
  my_tree = my_tree.fit(train[input_cols], train.target)

  ##  Evaluamos los ejemplos de testeo, y los comparamos con los reales para calcular el acierto, precisión, etc.

  # predecimos los ejemplos del conjunto de test

  test_pred = my_tree.predict(test[input_cols])

  # y los comparamos contra los "reales"
  aciertos_scikit = metrics.accuracy_score(test.target, test_pred) * 100
  print('Precisión de aciertos del algoritmo de scikit-learn:', f'{round(aciertos_scikit, 2)}%')
  evaluar_tasa_aciertos(aciertos_scikit, 'el algoritmo de scikit-learn')

  # veamos precisión, recuperación...
  if is_verbose :
    print()
    print(metrics.classification_report(test.target, test_pred))

  print()

  compare(aciertos_nuestro, aciertos_scikit)

  if show_tree :
    # # Imprimir el árbol de desición
    fig = plt.figure(figsize=(100,100))
    sk_tree.plot_tree(my_tree, filled=True, fontsize=12)
    plt.show()
    fig.savefig('./parte2.out.png', dpi=100)

if not is_multi :
  main()

def multi_test () :
  dataset = load_data(is_verbose)
  print()
  training_size = []
  acc_list = []
  acc_list_ours = []
  tree_sizes = []

  for x in  range(1,11) :
    print('EXPERIMENTO', x)
    i = (x*10)-1
    test_size = (100 - i)/100
    training_size.append(i)
    train, test = select_data(dataset, test_size)
    input_cols = dataset.columns[0:13]
    my_tree = sk_tree.DecisionTreeClassifier(criterion="entropy")
    my_tree = my_tree.fit(train[input_cols], train.target)
    test_pred = my_tree.predict(test[input_cols])
    tree_size = my_tree.tree_.node_count
    tree_sizes.append(tree_size)
    a = metrics.accuracy_score(test.target, test_pred)*100
    acc_list.append(a)

    print()
    print('   Algoritmo de scikit-learn')
    print('   Precisión de aciertos: ', f'{round(a,2)}%')
    print('   Cantidad de nodos', tree_size)
    if is_verbose :
      evaluar_tasa_aciertos(a, 'el algoritmo de scikit-learn')
    print()

    # Comparando con nuestro algoritmo
    tree = get_child_nodes(train)
    predictions = predict_all(test, tree)
    m, a = accuracy(predictions, test.target)
    acc_list_ours.append(a)
    tree_size = tree.getNodeCount()
    print('   Nuestra implementación de ID3')
    print('   Precisión de aciertos: ', f'{round(a,2)}%')
    print('   Cantidad de nodos', tree_size)
    if is_verbose :
      evaluar_tasa_aciertos(a, 'nuestra implementación del algoritmo ID3')
    print()

  tree_sizes.sort()

  a_prom = sum(acc_list) / len(acc_list)
  a_prom_ours = sum(acc_list_ours) / len(acc_list_ours)
  print('Tasa de aciertos promedio de scikit-learn: ', f'{round(a_prom, 2)}%')
  print('Tasa de aciertos promedio de nuestra implementación ', f'{round(a_prom_ours, 2)}%')
  compare(a_prom_ours, a_prom)

  if is_verbose :
    cm = 1/2.54
    plt.subplots(figsize=(30*cm, 10*cm))
    plt.plot(training_size, acc_list, '-')
    plt.xlabel('Porcentaje de entrenamiento')
    plt.ylabel('Aciertos [scikit]')
    plt.show()

if is_multi :
  multi_test()
