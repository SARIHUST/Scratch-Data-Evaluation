import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from utils.utils import load_churn_data

X_train, X_test, y_train, y_test = load_churn_data()
train_num = 800
X_train = X_train[:train_num]
y_train = y_train[:train_num]

# consider benchmark -- shapley values
svs = np.load('./ChurnPredict/svs_{}.npy'.format(train_num))
rank_sv = np.argsort(svs)

pca = PCA(n_components=10)
X_train_all = pca.fit_transform(X_train)
X_test_all = pca.transform(X_test)
LRPCA = LogisticRegression(max_iter=1000)
LRPCA.fit(X_train_all, y_train)
y_predict = LRPCA.predict(X_test_all)
print('all data points accuracy: {}'.format(accuracy_score(y_test, y_predict)))
print('all data points loss: {}'.format(log_loss(y_test, y_predict)))

groups = 5
for i in range(groups):
    idx = rank_sv[i * train_num // groups: (i + 1) * train_num // groups]
    X_train_i = X_train[idx]
    y_train_i = y_train[idx]
    pca = PCA(n_components=10)
    X_train_i = pca.fit_transform(X_train_i)
    X_test_i = pca.transform(X_test)
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train_i, y_train_i)
    y_predict = classifier.predict(X_test_i)
    print('sv{} data points accuracy: {}'.format(i, accuracy_score(y_test, y_predict)))
    print('sv{} data points loss: {}'.format(i, log_loss(y_test, y_predict)))

# consider distance
dist = np.load('./ChurnPredict/dist_{}.npy'.format(train_num)).flatten()
rank_dist = np.argsort(dist)

for i in range(groups):
    idx = rank_dist[i * train_num // groups: (i + 1) * train_num // groups]
    X_train_i = X_train[idx]
    y_train_i = y_train[idx]
    pca = PCA(n_components=10)
    X_train_i = pca.fit_transform(X_train_i)
    X_test_i = pca.transform(X_test)
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train_i, y_train_i)
    y_predict = classifier.predict(X_test_i)
    print('dist{} data points accuracy: {}'.format(i, accuracy_score(y_test, y_predict)))
    print('dist{} data points loss: {}'.format(i, log_loss(y_test, y_predict)))

# consider delta-distance
delta_dist = np.load('./ChurnPredict/delta_dist_{}.npy'.format(train_num))
rank_delta = np.argsort(delta_dist)

for i in range(groups):
    idx = rank_delta[i * train_num // groups: (i + 1) * train_num // groups]
    X_train_i = X_train[idx]
    y_train_i = y_train[idx]
    pca = PCA(n_components=10)
    X_train_i = pca.fit_transform(X_train_i)
    X_test_i = pca.transform(X_test)
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train_i, y_train_i)
    y_predict = classifier.predict(X_test_i)
    print('delta_dist{} data points accuracy: {}'.format(i, accuracy_score(y_test, y_predict)))
    print('delta_dist{} data points loss: {}'.format(i, log_loss(y_test, y_predict)))

show_num = 100
# show distance
svs = svs[:show_num]
dist = dist[:show_num]
delta_dist = delta_dist[:show_num]
idx = np.arange(1, show_num + 1)
s0 = plt.scatter(idx[y_train[:show_num]==0], dist[y_train[:show_num]==0], c='r', marker='x')
s1 = plt.scatter(idx[y_train[:show_num]==1], dist[y_train[:show_num]==1], c='b', marker='o')
plt.xlabel('data points')
plt.ylabel('distance')
plt.title('Distance of each data point to the hyper-plane')
plt.legend(handles=[s0, s1], labels=['type 0', 'type 1'])
plt.show()

# show shapley value
s0 = plt.scatter(idx[y_train[:show_num]==0], svs[y_train[:show_num]==0], c='r', marker='x')
s1 = plt.scatter(idx[y_train[:show_num]==1], svs[y_train[:show_num]==1], c='b', marker='o')
plt.xlabel('data points')
plt.ylabel('shapley value')
plt.title('Shapley Values of each data point')
plt.legend(handles=[s0, s1], labels=['type 0', 'type 1'])
plt.show()

# show converge
sv_it = np.load('ChurnPredict/sv_it_{}.npy'.format(train_num))
idx_it = np.arange(1, len(sv_it[0]) + 1)
plt.plot(idx_it, sv_it[0])
plt.xlabel('iterations')
plt.ylabel('shapley value')
plt.show()

# compare shapley value and distance
svs_ = svs / sum(svs) * 50
dist_ = dist[:show_num] / sum(dist[:show_num]) * 100 + 2.5
s0 = plt.scatter(idx, svs_, c='r', marker='x')
plt.plot(idx, svs_)
s1 = plt.scatter(idx, dist_, c='b', marker='o')
plt.plot(idx, dist_)
plt.xlabel('data points')
plt.title('shapley value vs distance to border')
plt.legend(handles=[s0, s1], labels=['sv', 'dist'])
plt.show()

# show delta distance
s0 = plt.scatter(idx, delta_dist[:show_num], c='r', marker='x')
plt.xlabel('data points')
plt.ylabel('delta distance')
plt.title('Delta distance by LOO strategy')
plt.show()

# compare distance and delta distance
dist_ = dist[:show_num] / sum(dist[:show_num]) * 50
delta_dist_ = delta_dist[:show_num] / sum(delta_dist[:show_num]) * 50 + 1.5
s0 = plt.scatter(idx, dist_, c='r', marker='x')
plt.plot(idx, dist_[:show_num])
s1 = plt.scatter(idx, delta_dist_, c='b', marker='o')
plt.plot(idx, delta_dist_[:show_num])
plt.xlabel('data points')
plt.title('distance vs delta distance')
plt.legend(handles=[s0, s1], labels=['dist', 'delta dist'])
plt.show()

# compare shapley value and delta distance
svs_ = svs / sum(svs) * 50
delta_dist_ = delta_dist[:show_num] / sum(delta_dist[:show_num]) * 50 + 2.5
s0 = plt.scatter(idx, svs_, c='r', marker='x')
plt.plot(idx, svs_)
s1 = plt.scatter(idx, delta_dist_, c='b', marker='o')
plt.plot(idx, delta_dist_[:show_num])
plt.xlabel('data points')
plt.title('shapley value vs delta distance')
plt.legend(handles=[s0, s1], labels=['sv', 'delta dist'])
plt.show()