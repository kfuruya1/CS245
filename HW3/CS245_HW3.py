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
