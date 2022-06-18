import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.decomposition import PCA

from utils.utils import shapley_values, load_churn_data

X_train, X_test, y_train, y_test = load_churn_data()

train_num = 800
X_train = X_train[:train_num]
y_train = y_train[:train_num]
# first model using all the features
LR = LogisticRegression(max_iter=1000)
LR.fit(X_train, y_train)
y_predict = LR.predict(X_test)
print(accuracy_score(y_test, y_predict))
print(log_loss(y_test, y_predict))

# second model uses PCA to get more important variables
pca = PCA(n_components=10)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
print(pca.explained_variance_ratio_)
LRPCA = LogisticRegression(max_iter=1000)
LRPCA.fit(X_train, y_train)
y_predict = LRPCA.predict(X_test)
print(accuracy_score(y_test, y_predict))
print(log_loss(y_test, y_predict))

# compute the distance
show_num = 100

w = LRPCA.coef_
b = LRPCA.intercept_
dist = np.array([abs(np.dot(w, x) + b) for x in X_train])
dist /= np.linalg.norm(w)
np.save('./ChurnPredict/dist_{}'.format(train_num), dist)

# compute LOO distance change
loo_dist = np.zeros(len(X_train))
for i in range(len(X_train)):
    if i % 100 == 0:
        print('computing on {}'.format(i))
    x = X_train[i]
    X_ = np.vstack((X_train[:i], X_train[i + 1:]))
    y_ = np.hstack((y_train[:i], y_train[i + 1:]))
    LR = LogisticRegression()
    LR.fit(X_, y_)
    w = LR.coef_
    b = LR.intercept_
    dist_i = abs(np.dot(w, x) + b)
    dist_i /= np.linalg.norm(w)
    loo_dist[i] = abs(dist[i] - dist_i)

np.save('ChurnPredict/delta_dist_{}'.format(train_num), loo_dist)
loo_dist = np.load('ChurnPredict/delta_dist_{}.npy'.format(train_num))

# compute shapley values
svs, sv_it = shapley_values(X_train[:train_num], y_train[:train_num], X_test, y_test, evaluate='loss', max_p=10)
np.save('ChurnPredict/svs_{}'.format(train_num), svs)
np.save('ChurnPredict/sv_it_{}'.format(train_num), sv_it)

svs = np.load('ChurnPredict/svs_{}.npy'.format(train_num))[:show_num]
sv_it = np.load('ChurnPredict/sv_it_{}.npy'.format(train_num))

# show distance
idx = np.arange(1, show_num + 1)
s0 = plt.scatter(idx[y_train[:show_num]==0], dist[:show_num][y_train[:show_num]==0], c='r', marker='x')
s1 = plt.scatter(idx[y_train[:show_num]==1], dist[:show_num][y_train[:show_num]==1], c='b', marker='o')
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
idx_it = np.arange(1, len(sv_it) + 1)
plt.plot(idx_it, sv_it)
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
s0 = plt.scatter(idx, loo_dist[:show_num], c='r', marker='x')
plt.xlabel('data points')
plt.ylabel('delta distance')
plt.title('Delta distance by LOO strategy')
plt.show()

# compare distance and delta distance
dist_ = dist[:show_num] / sum(dist[:show_num]) * 50
loo_dist_ = loo_dist[:show_num] / sum(loo_dist[:show_num]) * 50 + 1.5
s0 = plt.scatter(idx, dist_, c='r', marker='x')
plt.plot(idx, dist_[:show_num])
s1 = plt.scatter(idx, loo_dist_, c='b', marker='o')
plt.plot(idx, loo_dist_[:show_num])
plt.xlabel('data points')
plt.title('distance vs delta distance')
plt.legend(handles=[s0, s1], labels=['dist', 'delta dist'])
plt.show()

# compare shapley value and delta distance
svs_ = svs / sum(svs) * 50
loo_dist_ = loo_dist[:show_num] / sum(loo_dist[:show_num]) * 50 + 2.5
s0 = plt.scatter(idx, svs_, c='r', marker='x')
plt.plot(idx, svs_)
s1 = plt.scatter(idx, loo_dist_, c='b', marker='o')
plt.plot(idx, loo_dist_[:show_num])
plt.xlabel('data points')
plt.title('shapley value vs delta distance')
plt.legend(handles=[s0, s1], labels=['sv', 'delta dist'])
plt.show()