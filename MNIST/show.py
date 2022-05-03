'''
    This script shows the SV vs Distance of the data points.
'''

import matplotlib.pyplot as plt
import numpy as np

show_num = 100

svs = np.load('MNIST/calculation/shapley-values.npy')
dist = np.load('MNIST/calculation/distance.npy')
svs = svs[:show_num]
dist = dist[:, :show_num]
print(dist)
print(dist[0] == dist[1])

print(svs.shape, dist.shape)

svs = np.array([svs[i] for i in range(len(svs)) if dist[1][i] != dist[2][i]])
dists = np.array([dist[3][i] for i in range(len(svs)) if dist[1][i] != dist[2][i]])
svs = svs / sum(svs) * 100
dists = dists / sum(dists) * 100

idx = np.arange(1, show_num + 1)
s1 = plt.scatter(idx[dist[1] == dist[0]], svs[dist[1] == dist[0]], c='r', marker='x')
s2 = plt.scatter(idx[dist[1] != dist[0]], svs[dist[1] != dist[0]], c='b', marker='x')
s3 = plt.scatter(idx[dist[1] == dist[0]], dists[dist[1] == dist[0]], c='r', marker='o')
s4 = plt.scatter(idx[dist[1] != dist[0]], dists[dist[1] != dist[0]], c='b', marker='o')
plt.xlabel('data points')
plt.ylabel('proportion to total')
plt.title('Shapley Value vs Distance to border')
plt.legend(handles=[s1, s2, s3, s4], labels=['sv-c', 'sv-w', 'dist-c', 'dist-w'])
plt.show()