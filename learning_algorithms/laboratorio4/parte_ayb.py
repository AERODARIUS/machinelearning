
from common import load_data, text_to_nums, PCA_plot, minmax
from sklearn.decomposition import PCA
import numpy as np
from kmeans import kmeans
import random
from Elbow import elbow


dataset = load_data() 

#Descomentar uno de estos dos grupos de lineas

# columns_filtered = dataset.columns.drop(['Country', 'Region', "Agriculture","Industry","Service", "Climate"])
# elbowN = 6

columns_filtered = ["Net migration","Infant mortality (per 1000 births)", "GDP ($ per capita)", "Literacy (%)","Phones (per 1000)","Arable (%)","Crops (%)","Agriculture","Industry","Service"]
elbowN = 3


num_matrix = minmax(text_to_nums(dataset[columns_filtered])).to_numpy()
pca = PCA(n_components=2)
pca.fit(num_matrix)

def listClusters(clusters, dataset):
  res = []
  for cluster in clusters:
    resC = []
    i = 0
    for elem in dataset:
      for elemC in cluster:
        dist = np.linalg.norm(elem - elemC)
        if dist == 0:
          resC.append(i)
          break
      i += 1
    res.append(resC)
  return res

def printByColumn(list_clusters, dataset, ColumnName):
  i = 0

  for elem in list_clusters:
    regions_count = {}
    print('Cluster ', i, "\n")

    for index in elem:
      d = dataset.iloc[index][ColumnName].strip()

      if d in regions_count :
        regions_count[d] += 1
      else :
        regions_count[d] = 1

    if ColumnName == 'Country' :
      print(regions_count.keys())
    else :
      print(regions_count)
  
    print('\n\n\n')
    i += 1

def randomCentroides(cant, num_matrix):
  import math
  random.seed(42)
  res = []
  for i in range(cant):
    ran = math.floor(random.random()*227)
    res.append(num_matrix[ran])
  return np.array(res)

def clusterPerRegion(data):
  res = {}
  num_matrix = minmax(text_to_nums(data[columns_filtered])).to_numpy()

  for index, element in data.iterrows():
    d = element.Region.strip()
    if(d in res):
      res[d].append(num_matrix[index])
    else:
      res[d] = []
      res[d].append(num_matrix[index])
  aux = []
  for key in res:
    aux.append(res[key])

  return aux

def main() :

  PCA_plot(num_matrix, [], pca, labels = dataset.Country.to_numpy())

  elbow(num_matrix)

  centroides = randomCentroides(elbowN,num_matrix)
  clusters, centroides = kmeans(num_matrix, centroides)

  PCA_plot(clusters, centroides, pca, uruguay=num_matrix[215])
  
  
  list_clusters = listClusters(clusters, num_matrix)

  i = 0

  for c in centroides:
    print('centroide ', i, ':')
    print(c, '\n')
    i+=1

  print()
  printByColumn(list_clusters, dataset, 'Region')
  printByColumn(list_clusters, dataset, 'Country')

def separadosPorRegiones():
    clusterPRegion = clusterPerRegion(dataset)
    PCA_plot(clusterPRegion,randomCentroides(11,num_matrix) , pca, uruguay=num_matrix[215])
    list_clusters = listClusters(clusterPRegion, num_matrix)
    printByColumn(list_clusters, dataset, 'Region')

def coseno() :
  centroides = randomCentroides(elbowN,num_matrix)
  clusters, centroides = kmeans(num_matrix, centroides, isCoseno=True)

  PCA_plot(clusters, centroides, pca, uruguay=num_matrix[215])

