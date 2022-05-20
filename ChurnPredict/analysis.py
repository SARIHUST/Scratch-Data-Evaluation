import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.decomposition import PCA

from utils.utils import load_churn_data

X_train, X_test, y_train, y_test = load_churn_data()
X_train = X_train[:400]
y_train = y_train[:400]

# consider benchmark -- shapley values
svs = np.load('./ChurnPredict/svs_400.npy')
rank_sv = np.argsort(svs)[::-1]
X_train_sv_high = X_train[rank_sv[:200]]
y_train_sv_high = y_train[rank_sv[:200]]
X_train_sv_low = X_train[rank_sv[200:]]
y_train_sv_low = y_train[rank_sv[200:]]

pca = PCA(n_components=10)
X_train_all = pca.fit_transform(X_train)
X_test_all = pca.transform(X_test)
LRPCA = LogisticRegression(max_iter=1000)
LRPCA.fit(X_train_all, y_train)
y_predict = LRPCA.predict(X_test_all)
print('all 400 points accuracy: {}'.format(accuracy_score(y_test, y_predict)))
print('all 400 points loss: {}'.format(log_loss(y_test, y_predict)))

pca = PCA(n_components=10)
X_train_sv_high = pca.fit_transform(X_train_sv_high)
X_test_sv_high = pca.transform(X_test)
LRPCA = LogisticRegression(max_iter=1000)
LRPCA.fit(X_train_sv_high, y_train_sv_high)
y_predict = LRPCA.predict(X_test_sv_high)
print('sv high 200 points accuracy: {}'.format(accuracy_score(y_test, y_predict)))
print('sv high 200 points loss: {}'.format(log_loss(y_test, y_predict)))

pca = PCA(n_components=10)
X_train_sv_low = pca.fit_transform(X_train_sv_low)
X_test_sv_low = pca.transform(X_test)
LRPCA = LogisticRegression(max_iter=1000)
LRPCA.fit(X_train_sv_high, y_train_sv_low)
y_predict = LRPCA.predict(X_test_sv_low)
print('sv low 200 points accuracy: {}'.format(accuracy_score(y_test, y_predict)))
print('sv low 200 points loss: {}'.format(log_loss(y_test, y_predict)))

# consider distance
dist = np.load('./ChurnPredict/dist_400.npy').flatten()
rank_dist = np.argsort(dist)
X_train_near = X_train[rank_dist[:200]]
y_train_near = y_train[rank_dist[:200]]
X_train_far = X_train[rank_dist[200:]]
y_train_far = y_train[rank_dist[200:]]

pca = PCA(n_components=10)
X_train_near = pca.fit_transform(X_train_near)
X_test_near = pca.transform(X_test)
LRPCA = LogisticRegression(max_iter=1000)
LRPCA.fit(X_train_near, y_train_near)
y_predict = LRPCA.predict(X_test_near)
print('distance near 200 points accuracy: {}'.format(accuracy_score(y_test, y_predict)))
print('distance near 200 points loss: {}'.format(log_loss(y_test, y_predict)))

pca = PCA(n_components=10)
X_train_far = pca.fit_transform(X_train_far)
X_test_far = pca.transform(X_test)
LRPCA = LogisticRegression(max_iter=1000)
LRPCA.fit(X_train_far, y_train_far)
y_predict = LRPCA.predict(X_test_far)
print('distance far 200 points accuracy: {}'.format(accuracy_score(y_test, y_predict)))
print('distance far 200 points loss: {}'.format(log_loss(y_test, y_predict)))

# consider delta-distance
delta_dist = np.load('./ChurnPredict/delta_dist_400.npy')
rank_delta = np.argsort(delta_dist)
X_train_delta_high = X_train[rank_delta[200:]]
y_train_delta_high = y_train[rank_delta[200:]]
X_train_delta_low = X_train[rank_delta[:200]]
y_train_delta_low = y_train[rank_delta[:200]]

pca = PCA(n_components=10)
X_train_delta_high = pca.fit_transform(X_train_delta_high)
X_test_delta_high = pca.transform(X_test)
LRPCA = LogisticRegression(max_iter=1000)
LRPCA.fit(X_train_delta_high, y_train_delta_high)
y_predict = LRPCA.predict(X_test_delta_high)
print('delta distance high 200 points accuracy: {}'.format(accuracy_score(y_test, y_predict)))
print('delta distance high 200 points loss: {}'.format(log_loss(y_test, y_predict)))

pca = PCA(n_components=10)
X_train_delta_low = pca.fit_transform(X_train_delta_low)
X_test_delta_low = pca.transform(X_test)
LRPCA = LogisticRegression(max_iter=1000)
LRPCA.fit(X_train_delta_high, y_train_delta_low)
y_predict = LRPCA.predict(X_test_delta_low)
print('delta distance low 200 points accuracy: {}'.format(accuracy_score(y_test, y_predict)))
print('delta distance low 200 points loss: {}'.format(log_loss(y_test, y_predict)))