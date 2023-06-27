import matplotlib.pyplot as plt
from common import parse_args, load_data, select_data, evaluar_tasa_aciertos
from id3_nuestro import get_child_nodes, predict_all, accuracy

is_verbose, is_multi, show_tree = parse_args('Parte 1: Algoritmo ID3')

# Por ejemplo cuanod tengamos solo casos positivos o negativos, o pocas muestras
# Esto lo vamos a ver cuando lleguemos a los nodos finales
def main() :
  dataset = load_data(is_verbose)
  train, test = select_data(dataset, 0.2)

  tree = get_child_nodes(train)

  predictions = predict_all(test, tree)

  m, a = accuracy(predictions, test.target)

  if is_verbose :
    print()
    print('Predicciones')
    print('-------------')
    for p in m :
      print('ACIERTO' if p else 'FALLO')

  print()
  print('Precisión de aciertos del algoritmo:', f'{round(a,2)}%')
  evaluar_tasa_aciertos(a, 'nuestra implementación del algoritmo ID3')

  if show_tree :
    tree.display()

if not is_multi :
  main()

def multi_test () :
  dataset = load_data(is_verbose)
  training_size = []
  acc_list = []
  tree_sizes = []
  print()

  for x in  range(1,11) :
    print('EXPERIMENTO', x)
    i = (x * 10) - 1
    test_size = (100 - i) / 100
    training_size.append(i)
    train, test = select_data(dataset, test_size, is_verbose)
    tree = get_child_nodes(train)
    predictions = predict_all(test, tree)
    m, a = accuracy(predictions, test.target)
    print('Precisión de aciertos: ', f'{round(a,2)}%')
    tree_size = tree.getNodeCount()
    tree_sizes.append(tree_size)
    print('Cantidad de nodos', tree_size)
    if is_verbose :
      evaluar_tasa_aciertos(a, 'nuestra implementación del algoritmo ID3')
    print()
    acc_list.append(a)

    if show_tree :
      tree.display()
      print('================')
      print()
      print()
  
  a_prom = sum(acc_list) / len(acc_list)
  print('Tasa de aciertos promedio: ', f'{round(a_prom, 2)}%')
  
  tree_sizes.sort()

  if is_verbose :
    cm = 1/2.54
    plt.subplots(figsize=(30*cm, 10*cm))
    plt.plot(training_size, acc_list, '-')
    plt.xlabel('Porcentaje de entrenamiento')
    plt.ylabel('Aciertos')
    plt.show()

if is_multi :
  multi_test()
