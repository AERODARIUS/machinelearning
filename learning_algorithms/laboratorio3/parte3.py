# coding=utf-8
from common import load_data, parse_args, select_data_casero, get_stats, sprlit_n, names_to_nums, text_to_nums, normalize
import math
from pandas import concat

is_verbose = parse_args('Laboratorio 3')
names_to_nums.pop('term_deposit')
dataset = text_to_nums(load_data().replace(names_to_nums), True)
train, test = select_data_casero(dataset, 0.2)

def count_train_data (train_data, attribute, value, td) :
  return train_data[(train_data[attribute] == value) & (train_data.term_deposit == td)].term_deposit.size

def normalP(x, max_val) :
  max = x - max_val # trasladar el pico al valor maximo
  return (math.exp(-(max**2)/2) / (math.sqrt(2*math.pi)))

# Clasificador Simple 
def entrenar(train_data, attributes, is_custom) :
  yes_rows =  train_data.term_deposit.value_counts().yes
  no_rows =  train_data.term_deposit.value_counts().no
  defProb = {}
  probabilities = {}
  normal_dist = {'age_range', 'sibilings', 'marital', 'education', 'default', 'housing', 'loan', 'contact'}

  if is_verbose :
    print('Distribución de los datos')

  for attribute in attributes :
    if is_verbose :
      print('-', attribute)
    
    value_counts = train_data[attribute].value_counts()
    max_val = value_counts.idxmax()
    probabilities[attribute] = {}
    m = value_counts.size # cantidad de valores posibles
    p = (1 / m) # por defecto consideramos los valores son equiprobables

    for value, count in value_counts.iteritems() :
      if is_verbose :
        print('   ', value, count)

      if is_custom and attribute in normal_dist :
        p = normalP(value, max_val)

      mp = m * p

      e_yes = count_train_data(train_data, attribute, value, 'yes')
      e_no = count_train_data(train_data, attribute, value, 'no')

      probabilities[attribute][value] = {
        'yes': (e_yes + mp) / (yes_rows + m), # P(value|yes) = (e + m * p) / (n + m)
        'no': (e_no + mp) / (no_rows + m), # P(value|no) = (e + m * p) / (n + m)
        'total': count,
      }
    defProb[attribute] = {
      'yes': 1 / (yes_rows + m),
      'no': 1 / (no_rows + m),
      'total': 0
    }

  return probabilities, defProb


def calcular_P(instancia, probabilities, defProb, train_data, attributes) :
  total_rows = train_data.term_deposit.size
  yes_rows =  train_data.term_deposit.value_counts().yes
  no_rows =  train_data.term_deposit.value_counts().no
  P_yes = yes_rows / total_rows # P(yes)
  P_no = no_rows / total_rows # P(no)
  P_i_yes = P_yes
  P_i_no = P_no
  P_i = 1 # P(instance) = P(atribute_1) * p(attribute_2) * ... * P(attribute_n)

  for attribute in attributes :
    value = instancia[attribute]
    P_attr_val = probabilities[attribute][value] if value in probabilities[attribute] else defProb[attribute]
    # Productoria de la probabilida de cada atributo
    # y la la probabilidad de que sea yes/no
    P_i_yes *= P_attr_val['yes']
    P_i_no *= P_attr_val['no']
    P_i *= (P_attr_val['total'] / total_rows)

  P_i_y_n = None # P(instancia|yes) or P(instancia|no)
  y_n = None

  # Hipótesis más probable
  if P_i_yes < P_i_no :
    P_i_y_n = P_i_no
    y_n = 'no'
  else :
    P_i_y_n = P_i_yes
    y_n = 'yes'
  
  # Seguridad de la hipótesis más probable
  # Calculando Bayes: P(D/h) * P(h) / P(D)
  # Para calcular P(D) se puede usar los P(D/h) y P(D/no_h) que estimamos, ya que P(D) = P(D/h) + P(D/no_h), osea P_i_no + P_i_yes
  # Luego P(D/h) * P(h) = P(D/attributo_1) * P(D/attributo_2) * ... * P(D/attributo_n) * P(h), osea  P_i_y_n
  seguridad = round((P_i_y_n / (P_i_no + P_i_yes))*100, 2)


  return y_n, P_i_y_n, seguridad


def get_predictions(test_data, train_data, probabilities, defProb, attributes) :
  predictions = []

  if is_verbose :
    print('Probando el algoritmo')

  for row in test_data.itertuples() :
    dictio = dict(row._asdict())
    predict, prob, seguiridad = calcular_P(dictio, probabilities, defProb, train_data, attributes)

    if is_verbose :
      extra_tab = '\t' if predict == 'no' else ''
      print(f'predición: {predict}, {extra_tab} \tvalor real: {dictio["term_deposit"]}, \tP(instancia/valor): {prob},   \tconfianza de P: {seguiridad}\n')

    predictions.append(predict)

  return predictions, seguiridad

def get_model(train_data1, attributes, is_custom = False) :
  probabilities, defProb = entrenar(train_data1, attributes, is_custom)

  def model(train_data2) :
    predictions, seguridad = get_predictions(train_data2, train_data1, probabilities, defProb, attributes)
    sats = get_stats(train_data2.term_deposit.values.ravel(), predictions, 1)
    #  accuracy, precisión, recall y medida-F
    return list(map(lambda x : round(x*100, 2), sats)), seguridad

  return model

def experimento0(train_data, validate_data) :
  attributes = train.keys().drop('term_deposit')
  model = get_model(train_data, attributes)

  # Validación
  stats, seguridad = model(validate_data)

  print('Estadeisticas de la iteración')
  print(stats, '- accuracy, precision, recall, medidaF')
  print()

  return stats, model


# Sin las columnas "pdays" y "poutcome" y ver si mejora el resultado
def experimento1y2(train_data, validate_data, to_drop) :
  attributes = train.keys().drop('term_deposit')
  model = get_model(train_data, attributes.drop(to_drop))
  stats, seguridad = model(validate_data)

  print('Estadeisticas de la iteración')
  print(stats, '- accuracy, precision, recall, medidaF')
  print()

  return stats, model

def experimento3(train_data, validate_data) :
  # Naive bayes asume que cada atributo es independiente del otro
  # por lo que tiene sentido probar esto
  attributes = train.keys().drop('term_deposit')
  model = get_model(train_data, attributes, True)

  # Validación
  stats, seguridad = model(validate_data)

  print('Estadeisticas de la iteración')
  print(stats, '- accuracy, precision, recall, medidaF')
  print()

  return stats, model

def get_stats_prom(stats_list) :
  stats_len = len(stats_list)
  accuracyM = 1
  precisionM = 1
  recallM = 1
  medidaFM = 1

  for stats in stats_list :
    accuracy, precision, recall, medidaF = stats
    if accuracy > 0 and precision > 0 and medidaF > 0 and medidaF > 0 :
      accuracyM += accuracy
      precisionM += precision
      recallM += recall
      medidaFM += medidaF
    else :
      stats_len -= 1

  def prom(x) :
    return round(x/ stats_len, 2) if stats_len > 0 else -1

  return prom(accuracyM), prom(precisionM), prom(recallM), prom(medidaFM)

def experiment(method, train_data_n, description, to_drop = None) :
  stats_list = []

  print(description)
  print('------------------------------------------------------------------------')
  for ind, validate in enumerate(train_data_n) :
    train_data_list = train_data_n.copy()
    train_data_list.pop(ind)
    train_data = concat(train_data_list)
    stats, model = method(train_data, validate) if to_drop == None else method(train_data, validate, to_drop)
    stats_list.append(stats)

  stats_prom = get_stats_prom(stats_list)
  print()
  print('Estadeisticas promedio')
  print(stats_prom, '- accuracy, precision, recall, medidaF')
  print()

  return stats_prom, model

def main() :
  train_size = 10
  train_data_n = sprlit_n(train, train_size)
  all_names = []
  all_stats = []
  all_models = []
  
  print('========================================================================')
  stats, model = experiment(experimento0, train_data_n, 'Experimento 0: Sin quitar columnas y todos los valores equiprobables')
  print('------------------------------------------------------------------------')
  all_stats.append(stats)
  all_names.append('Sin quitar columnas y todos los valores equiprobables')
  all_models.append(model)

  print('========================================================================')
  print('Experimento 1: Probando quitar atributos aparentemente correlacionados')
  print('------------------------------------------------------------------------')
  for to_drop in ['pdays', 'poutcome'] :
    stats, model  = experiment(experimento1y2, train_data_n, f'Quitando el atributo "{to_drop}"', to_drop)
    all_stats.append(stats)
    all_names.append(f'Quitando el atributo "{to_drop}"')
    all_models.append(model)

  print('========================================================================')
  print('Experimento 2: Probando quitar atributos con poca dispersión')
  print('------------------------------------------------------------------------')
  for to_drop in [
              'default',
              'loan',
              'campaign',
              'pdays',
              'previous',
              'poutcome',
              ['default', 'previous'],
            ] :
    stats, model = experiment(experimento1y2, train_data_n, f'Quitando el/los atributo/s "{to_drop}"', to_drop)
    all_stats.append(stats)
    all_names.append(f'Quitando el/los atributo/s "{to_drop}"')
    all_models.append(model)

  print('========================================================================')
  stats, model = experiment(experimento3, train_data_n, 'Experimento 3: Probando usar distribuciones diferentes a la equiprobable')
  print('------------------------------------------------------------------------')
  all_stats.append(stats)
  all_names.append('Distribuciones diferentes a la equiprobable')
  all_models.append(model)

  best_stats = {
    'accuracy': 0,
    'precision': 0,
    'recall': 0,
    'medidaF': 0
  }
  best_name = ''
  best_model = None

  for ind, s in enumerate(all_stats) :
    accuracy, precision, recall, medidaF = s

    if medidaF > best_stats['medidaF'] :
      best_stats['accuracy'] = accuracy
      best_stats['precision'] = precision
      best_stats['recall'] = recall
      best_stats['medidaF'] = medidaF
      best_name = all_names[ind]
      best_model = all_models[ind]

  
  print('========================================================================')
  print('========================================================================')
  print()
  print('Las mejores estadísticas fueron:')
  print(best_name, best_stats)

  if abs(best_stats['medidaF'] - all_stats[0][3]) < 1 :
    print('------------------------------------------------------------------------')
    print('La mejora no fue lo suficientemente grande para elegir este modelo.')
    print('Decidimos quedarnos con el primer modelo sin modificaciones')
    print(all_stats[0])
    accuracy, precision, recall, medidaF = all_stats[0]
    best_stats['accuracy'] = accuracy
    best_stats['precision'] = precision
    best_stats['recall'] = recall
    best_stats['medidaF'] = medidaF
    best_stats = all_names[0]
    best_model = all_models[0]


  stats, seguridad = best_model(test)
  print('Estadeisticas del mejor modelo eleigdo sobre los datos de testing')
  print(stats, seguridad, '- accuracy, precision, recall, medidaF, seguridad')
  print()

main()
