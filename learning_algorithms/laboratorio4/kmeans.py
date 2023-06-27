from math import isclose
from common import coseno_dist
import numpy as np

def isnan(dataset) :
  print('NAN')
  print(dataset[dataset.isna().all(axis=1)])

normas = []

# Evaleua si la norma entre un centroide y el anterior es menor que un epsilon
# Es la condición de convergencia del algoritmo
def compare_c(c1, c2, isCoseno) :
  epsilon = 0.01

  if isCoseno : 
    dist = coseno_dist(c1, c2)
  else :
    dist = np.linalg.norm(c1 - c2)

  print(dist)

  return dist < epsilon

def converged(c1l, c2l, n, isCoseno) :
  print('Distancia centroides:')
  distances = [compare_c(c1l[k], c2l[k], isCoseno) for k in range(n)]
  print('--------------------------')
  return np.all(distances)

def init_clusters(n_clusters, n_attributes) :
  clusters =  []

  for _ in range(n_clusters) :
    cluster = np.ndarray([0, n_attributes]) # lista de instancias vacía
    clusters.append(cluster)
  
  return clusters


def not_empty(clusters) :
  for _, c in  enumerate(clusters) :
    if c.size == 0 :
      return True
  
  return False

def kmeans(dataset, centroides_ini, isCoseno=False) :
  print('Centroides iniciales:')
  print(centroides_ini)
  print()
  print()
  n_attributes = dataset.shape[1]
  n_clusters = centroides_ini.shape[0]
  centroides = centroides_ini.copy() - 1 # Inicialización para cumplir condición del while la primera vez
  new_centroides = centroides_ini.copy()
  clusters =  init_clusters(n_clusters, n_attributes)

  while not converged(centroides, new_centroides, n_clusters, isCoseno) : # and not_empty(clusters) :
    centroides = new_centroides.copy() # es deep copy, testeado
    clusters =  init_clusters(n_clusters, n_attributes)

    # Asignar cada instancia al centroide más cercano
    for instance in dataset :
      best_centr = None
      best_dist = np.inf

      # Elegir el cnetroide que esté más cerca de esta instancia
      for ind, c in enumerate(centroides) :
        if isCoseno :
          dist = coseno_dist(instance, c)
        else :
          dist = np.linalg.norm(instance - c)


        if dist < best_dist :
          best_dist = dist
          best_centr = ind

      # Guardar la isntancia en el cluster cuyo centroide es el más cercano
      clusters[best_centr] = np.append(clusters[best_centr], [instance], axis=0)

    # Recalcular los centroides de cada cluster
    # El centroide se calcula como la media de las instancias
    # print(clusters[0].shape, clusters[1].shape)
    for ind, c in  enumerate(clusters) :
      Nk = c.shape[0]
      xq_sum = np.add.reduce(c)
      new_centroides[ind] = xq_sum / Nk # xq_prom
      
  print()
  print()
  print('Centroides finales:')
  print(new_centroides)
  print()
  print()

  return clusters, new_centroides
