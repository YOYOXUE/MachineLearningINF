

import numpy as np
from sklearn import mixture
# %matplotlib inline
import matplotlib.pyplot as plt

with open("clusters.txt") as f:
    d = f.readlines()
n = len(d)  # num of total data
dim = 2  # dimension of data
data = np.zeros((n,dim), dtype='float64')  # data
for i in range(n):
    data_ = d[i].split(',')
    data[i][0] = float(data_[0])
    data[i][1] = float(data_[1])
# print(data)
m=mixture.GaussianMixture(n_components=3)
m.fit(data)
labels = m.predict(data)
plt.scatter(data[:, 0], data[:, 1], c=labels, s=40, cmap='viridis')
plt.show()
covariances = m.covariances_
means=m.means_
print("Means:\n",means)
print("Covariances:\n",covariances)
print("Amplitude:")
for i in covariances:
    print(1.0 / np.linalg.det(i))
