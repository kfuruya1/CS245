from sklearn.cluster import KMeans
import numpy as np

data_points = np.array([[9,5], [5,10], [10,11],
                        [4, 9], [14, 2], [4, 1],
                        [9, 9], [3, 12], [19, 6],
                        [17, 13], [13, 15]])

kmeans = KMeans(n_clusters = 2, init=([[10,13], [7,10]]), n_init = 'auto').fit(data_points)

kmeans.cluster_centers_

kmeans.labels_

data_points2 = np.array([[0, 0], [1, 0], [-1, 0], [0, -1], [2, 2], [-1, -1], [1, -1], [0,1]])
#data_points2 = np.array([[0, 0], [-1, -2], [1, 1], [-1, 1], [1, -1],
#                         [4, 0], [3, 0], [4, -2], [5, 1], [3, -1], [3, 4],
#                         [4, 4], [5, 4], [3, 5], [4, 6], [5, 5],
#                         [-1, 6], [-1, 4]])

kmeans_rand = KMeans(n_clusters = 2, init='random', n_init='auto').fit(data_points2)
kmeans_plus = KMeans(n_clusters = 2, init='k-means++', n_init='auto').fit(data_points2)

print("random centers ",kmeans_rand.cluster_centers_)
print(kmeans_rand.labels_)
print("plus centers ", kmeans_plus.cluster_centers_)
print(kmeans_plus.labels_)

"""[[-0.66666667 -0.33333333]

 [ 4.75        4.75      ]

 [ 3.16666667  0.        ]]

[0 0 2 0 2 2 2 2 2 1 1 1 1]

[[ 4.75  4.75]

 [ 3.4  -0.4 ]

 [ 0.    0.25]]

[2 2 2 2 1 1 1 1 1 0 0 0 0]
"""

from matplotlib import pyplot as plt

x = [item[0] for item in data_points2]
y = [item[1] for item in data_points2]
plt.scatter(x, y)

#data_points2 = np.array([[0, 0], [-1, -2], [1, 1], [-1, 1], [1, -1],
#                         [4, 0], [3, 0], [4, -2], [5, 1], [3, -1], [3, 4],
##                         [4, 4], [5, 4], [3, 5], [4, 6], [5, 5],
  #                       [-1, 6], [-1, 4]])
#[0 0 0 0 0 0 0 0 1 0 1 1 1 1 1 1 2 2]
#[1 1 1 1 1 2 2 2 2 2 0 0 0 0 0 0 0 0]


#random centers  [[ 1.55555556 -0.44444444]
# [ 4.14285714  4.14285714]
# [-1.          5.        ]]
#[0 0 0 0 0 0 0 0 1 0 1 1 1 1 1 1 2 2]
#plus centers  [[ 2.7500000e+00  4.7500000e+00]
# [-4.4408921e-16 -2.0000000e-01]
# [ 3.8000000e+00 -4.0000000e-01]]
#[1 1 1 1 1 2 2 2 2 2 0 0 0 0 0 0 0 0]

#data_points2 = np.array([[0, 0], [1, 0], [-1, 0], [0, -1], [2, 2], [-1, -1], [1, -1], [0,1]])
#[1 0 1 1 0 1 1 0]
#[0 0 0 0 1 0 0 0]



c1_rand = [[0, 0], [-1, 0], [0, -1], [-1, -1], [1, -1], [0, 1]]#[[0, 0], [-1, -2], [1, 1], [-1, 1], [1, -1], [4, 0], [3, 0], [4, -2], [3,-1]]
c2_rand = [[1, 0], [2, 2], [0, 1]] #[[5, 1], [3, 4], [4, 4], [5, 4], [3, 5], [4, 6], [5, 5]]
#c3_rand = [[-1, 6], [-1, 4]]

c1_plus = [[0, 0], [1, 0], [-1, 0], [0, -1], [-1, -1], [1, -1], [0,1]] #[[0, 0], [-1, -2], [1, 1], [-1, 1], [1, -1]]
c2_plus = [[2, 2]] #[[4, 0], [3, 0], [4, -2], [5, 1], [3, -1]]
#c3_plus = [[3, 4], [4, 4], [5, 4], [3, 5], [4, 6], [5, 5], [-1, 6], [-1, 4]]

x1_rand = [item[0] for item in c1_rand]
y1_rand = [item[1] for item in c1_rand]
x2_rand = [item[0] for item in c2_rand]
y2_rand = [item[1] for item in c2_rand]
#x3_rand = [item[0] for item in c3_rand]
#y3_rand = [item[1] for item in c3_rand]
plt.scatter(x1_rand, y1_rand, color='red')
plt.scatter(x2_rand, y2_rand, color='blue')
#plt.scatter(x3_rand, y3_rand, color='green')
plt.show()

x1_plus = [item[0] for item in c1_plus]
y1_plus = [item[1] for item in c1_plus]
x2_plus = [item[0] for item in c2_plus]
y2_plus = [item[1] for item in c2_plus]
#x3_plus = [item[0] for item in c3_plus]
#y3_plus = [item[1] for item in c3_plus]
plt.scatter(x1_plus, y1_plus, color='red')
plt.scatter(x2_plus, y2_plus, color='blue')
#plt.scatter(x3_plus, y3_plus, color='green')
plt.show()

#2
from sklearn.cluster import AgglomerativeClustering

d1_data = np.array([10, 2, 8, 12, 13, 15]).reshape(-1, 1)
clustering = AgglomerativeClustering().fit(d1_data)

clustering.labels_

#3
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

data3 = np.array([[10,-20], [5, 12],
                  [24, 22], [-2, -7],
                  [-3, -10], [12, -20],
                   [8, 17], [9, 11],
                    [18, 20], [8, 2],
                     [20, -10]])

data3 = StandardScaler().fit_transform(data3)


db_clust = DBSCAN(eps = 1, min_samples = 2).fit(data3)

print(data3)

from sklearn.metrics.pairwise import euclidean_distances

temp = euclidean_distances(data3, data3)
print(temp)

import pandas as pd

col_names = ['[10,-20]', '[5, 12]',
                  '[24, 22]', '[-2, -7]',
                  '[-3, -10]', '[12, -20]',
                   '[8, 17]', '[9, 11]',
                    '[18, 20]', '[8, 2]',
                     '[20, -10]']
index_names = ['[10,-20]', '[5, 12]',
                  '[24, 22]', '[-2, -7]',
                  '[-3, -10]', '[12, -20]',
                   '[8, 17]', '[9, 11]',
                    '[18, 20]', '[8, 2]',
                     '[20, -10]']

df = pd.DataFrame(temp, columns = col_names, index = index_names)
print(df)

db_clust.labels_

db_clust.core_sample_indices_

db_clust2 = DBSCAN(eps = 2, min_samples = 2).fit(data3)
db_clust2.labels_

db_clust2.core_sample_indices_

db_clust3 = DBSCAN(eps = 0.5, min_samples = 2).fit(data3)
db_clust3.labels_

db_clust3.core_sample_indices_

x_db = [item[0] for item in data3]
y_db = [item[1] for item in data3]
plt.scatter(x_db, y_db)

from sklearn.neighbors import LocalOutlierFactor

clf = LocalOutlierFactor(n_neighbors = 5)
y_pred = clf.fit_predict(data3)
x_scores = clf.negative_outlier_factor_

plt.scatter(data3[:, 0], data3[:, 1], color = "k", s= 3.0)
radius = (x_scores.max() - x_scores) / (x_scores.max() - x_scores.min())
scatter = plt.scatter(data3[:, 0], data3[:, 1], s=1000 * radius, edgecolors = "r", facecolors="None")
plt.axis("tight")
plt.xlim((-2, 2))
plt.ylim((-2, 2))
plt.show()

print(x_scores)
print(y_pred)

from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
from mlxtend.plotting import plot_decision_regions

svm = SVC(C=1, kernel = 'linear')
X = np.array([[3, 3], [2, 3], [2,2], [4,1], [4,2], [5,1]])
y = np.array([1,1,1,-1, -1, -1])
svm2 = SVC(kernel='rbf', gamma = 0.1, C=0.5)

svm.fit(X, y)
svm2.fit(X, y)
plot_decision_regions(X, y, clf=svm2, legend = 2)
plt.show
