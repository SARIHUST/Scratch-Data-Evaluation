import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from utils.utils import load_churn_data

X_train, X_test, y_train, y_test = load_churn_data(refine=1)
num = len(X_train)

pca = PCA(n_components=10)
X_train_all = pca.fit_transform(X_train)
X_test_all = pca.transform(X_test)
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train_all, y_train)
y_predict = classifier.predict(X_test_all)
print('all data points accuracy: {}'.format(accuracy_score(y_test, y_predict)))
print('all data points loss: {}'.format(log_loss(y_test, y_predict)))

# compute distance to the border for each data point
w = classifier.coef_
b = classifier.intercept_
dist = np.array([abs(np.dot(w, x) + b).item() for x in X_train_all])
dist /= np.linalg.norm(w)

# show the difference using different groups
groups = 6

rank_dist = np.argsort(dist.flatten())
dist_idxs = []
for i in range(groups):
    dist_idx = rank_dist[i * num // groups: (i + 1) * num // groups]
    dist_idxs.append(dist_idx)

for i, idx in enumerate(dist_idxs):
    Xi = X_train_all[idx]
    yi = y_train[idx]
    classifier.fit(Xi, yi)
    y_predict = classifier.predict(X_test_all)
    print('dist{} data points accuracy: {}'.format(i, accuracy_score(y_test, y_predict)))
    print('dist{} data points loss: {}'.format(i, log_loss(y_test, y_predict)))
    print('dist{} 0-labeled: {}, 1-labeled: {}'.format(i, sum(yi == 0), sum(yi == 1)))

delta_dist = np.zeros(num)
for i in range(num):
    if i % 1000 == 0:
        print('computing on {}'.format(i))
    x = X_train_all[i]
    X_ = np.vstack((X_train_all[:i], X_train_all[i + 1:]))
    y_ = np.hstack((y_train[:i], y_train[i + 1:]))
    LR = LogisticRegression()
    LR.fit(X_, y_)
    w = LR.coef_
    b = LR.intercept_
    dist_i = abs(np.dot(w, x) + b)
    dist_i /= np.linalg.norm(w)
    delta_dist[i] = abs(dist[i] - dist_i)
np.save('./ChurnPredict/delta_dist', delta_dist)

delta_dist = np.load('./ChurnPredict/delta_dist.npy')
rank_delta_dist = np.argsort(delta_dist.flatten())
delta_dist_idxs = []
for i in range(groups):
    delta_dist_idx = rank_delta_dist[i * num // groups: (i + 1) * num // groups]
    delta_dist_idxs.append(delta_dist_idx)

for i, idx in enumerate(delta_dist_idxs):
    Xi = X_train_all[idx]
    yi = y_train[idx]
    classifier.fit(Xi, yi)
    y_predict = classifier.predict(X_test_all)
    print('delta dist{} data points accuracy: {}'.format(i, accuracy_score(y_test, y_predict)))
    print('delta dist{} data points loss: {}'.format(i, log_loss(y_test, y_predict)))
    print('delta dist{} 0-labeled: {}, 1-labeled: {}'.format(i, sum(yi == 0), sum(yi == 1)))

for d in dist_idxs:
    for dd in delta_dist_idxs:                                                                                   
        print(sum(np.in1d(d, dd)), end=' ')
    print()

mat = np.vstack((dist, delta_dist))
print(mat)
rho = np.corrcoef(mat)
print(rho)

cov = np.cov(mat)
print(cov)

exit()
# visualize the relation between distance and delta distance
show_num = 100
show_dist = dist[:show_num]
show_dist = show_dist / sum(show_dist) * 100
show_delta_dist = delta_dist[:show_num]
show_delta_dist = show_delta_dist / sum(show_delta_dist) * 100 + 2.5
idx = np.arange(1, show_num + 1)
sd = plt.scatter(idx, show_dist, c='r', marker='x')
plt.plot(idx, show_dist)
sdd = plt.scatter(idx, show_delta_dist, c='b', marker='o')
plt.plot(idx, show_delta_dist)
plt.xlabel('data points')
plt.title('distance vs delta distance')
plt.legend(handles=[sd, sdd], labels=['dist', 'delta dist'])
plt.show()