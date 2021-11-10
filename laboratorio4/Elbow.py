from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from kmeans import kmeans

def elbow(dataset):
    distortions = []
    K = range(2, 10)
    centroides2 = np.array([dataset[0], dataset[208]])
    centroides3 = np.array([dataset[0], dataset[208], dataset[220]])
    centroides4 = np.array([dataset[80], dataset[208], dataset[220], dataset[189]])
    centroides5 = np.array([dataset[80], dataset[208], dataset[220], dataset[189], dataset[219]])
    centroides6 = np.array([dataset[80], dataset[208], dataset[220], dataset[189], dataset[219], dataset[210]])
    centroides7 = np.array([dataset[80], dataset[208], dataset[220], dataset[189], dataset[219], dataset[210], dataset[125]])
    centroides8 = np.array([dataset[80], dataset[208], dataset[220], dataset[189], dataset[219], dataset[210], dataset[125], dataset[65]])
    centroides9 = np.array([dataset[80], dataset[208], dataset[220], dataset[189], dataset[219], dataset[210], dataset[125], dataset[65], dataset[0]])
    centroidesList = [centroides2,centroides3, centroides4, centroides5, centroides6, centroides7, centroides8, centroides9]

    cont = 0

    for k in K:
        print('======================================================================================')
        print('Elbow, probando con k =', k)
        print()
        centroides = centroidesList[cont]
        clusters, centroides = kmeans(dataset, centroides)
        # Building and fitting the model
        distortions.append(sum(np.min(cdist(dataset, centroides,'euclidean'), axis=1)) / dataset.shape[0])
        cont += 1
    
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method using Distortion')
    plt.show()



