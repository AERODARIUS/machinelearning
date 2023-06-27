from common import entropy, load_data, load_data, text_to_nums, parse_args
import scipy.stats as st
from scipy.optimize import brute, differential_evolution
from scipy.stats import binom
import warnings
import numpy as np
# distribución de clase
# En este caso solo hay dos claes, `yes` y `no`
# Entonces hay que agrupar las filas que son `yes` y filas que son `no`
# Luego ver para cada atributo como se distribuyen los valores en esa clase

def func(free_params, *args) :
    dist, x = args
    ll = -np.log(dist.pmf(x, *free_params)).sum()

    if np.isnan(ll) :
        ll = np.inf

    return ll

def fit_discrete(dist, x, bounds, optimizer=brute) :
    with warnings.catch_warnings(): 
        warnings.simplefilter("ignore")  
        return optimizer(func, bounds, args=(dist, x))


def print_best_distribution(values_count, attribute, value, is_verbose) :
  with warnings.catch_warnings() : 
    warnings.simplefilter("ignore")  
    dist_continuas = ['uniform', 'norm', 'norminvgauss', 'pareto', 'exponweib', 'weibull_max', 'weibull_min']
    dist_discretas = ['bernoulli', 'binom', 'poisson']
    dist_names = {
      'uniform': 'uniforme',
      'norm': 'Normal (Gaussiana)',
      'norminvgauss': 'Gaussiana inversa',
      'bernoulli': 'Bernoulli',
      'binom': 'Binomial',
      'poisson': 'Poisson',
      'pareto': 'Pareto',
      'exponweib': 'Weibull exponencial',
      'weibull_max': 'Weibull máxima',
      'weibull_min': 'Weibull mínima',
      'unkown': 'Desconocida'
    }

    dist_results = []
    params = {}

    # Para distribuciones continuas
    for dist_name in dist_continuas :
      dist = getattr(st, dist_name)
      param = dist.fit(values_count)
      params[dist_name] = param
      # Aplicando el test de Kolmogorov-Smirnov
      D, p = st.kstest(values_count, dist_name, args=param)
      dist_results.append((dist_name, p))

    # Para distribuciones discretas
    for dist_name in dist_discretas :
      dist = getattr(st, dist_name)
      bounds = [(0, 100), (0, 1)]
      D, p = fit_discrete(dist, values_count, bounds)
      dist_results.append((dist_name, p))

    # La mejor distribución, no necesariamente es una buena aproximación
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    best_p = round(best_p * 100, 2)

    if best_p > 90 :
      if is_verbose :
        if value == None :
          print(f'# La distribución de "{attribute}" se aproxima más a {dist_names[best_dist]}, en un {best_p}%')
        else :
          print(f'La distribución de "term_deposit" fijando "{attribute}"="{value}" se aproxima más a una {dist_names[best_dist]}, en un {best_p}%')
    else :
      if is_verbose :
        if value == None :
          print(f'# Para "{attribute}" no pudimos identificar ninguna distribución en particular')
        else :
          print(f'Fijando "{attribute}"="{value}", "term_deposit" no pudimos identificar ninguna distribución en particular')
      best_dist = 'unkown'
      best_p = None

    return dist_names[best_dist], best_p



is_verbose = parse_args('Laboratorio 3')

def main():
  setRaw = load_data()

  # Transformar textos a valores numéricos
  set = text_to_nums(setRaw)
  if is_verbose :
    print(set)

  keys = set.keys()

  # Verificar la independencia entre columnas
  print('Buscando correlación entre atributos')
  for k in set.keys() :
    keys = keys.drop(k)
    for l in keys :
      correlation = set[k].corr(set[l])
      if correlation > 0.8 :
        print(f'"{k}" y "{l}" podrían estar correlacionados, existe un {round(correlation*100, 2)}% de correlación\n')

  if is_verbose :
    print('Analizamos la distribución y entropía de los datos')

  dist_map = {}

  for k in setRaw.keys().drop('term_deposit') :
    value_counts = setRaw[k].value_counts()

    # Puede servir para decidir cómo tratar los unknown
    print_best_distribution(value_counts, k, None, True)

    for v, count in value_counts.iteritems() :
      if is_verbose :
        print()
        print(f'Atributo: {k}, Valor: {v}')
        print("Entropía:", entropy(setRaw[setRaw[k] == v]))
        print('Cantidad:', count)

      vc_filter = setRaw[setRaw[k] == v].term_deposit.value_counts()
      # Para ver cómo se clasifican(yes/no) en ese atributo-valor
      best_dist, _ = print_best_distribution(vc_filter, k, v, is_verbose)

      if best_dist not in dist_map :
        dist_map[best_dist] = 1
      else :
        dist_map[best_dist] += 1

    if is_verbose :
      print("\n------------------------------------------------\n")
main()