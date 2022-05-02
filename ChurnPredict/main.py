import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.decomposition import PCA

from utils.utils import shapley_values, load_churn_data

X_train, X_test, y_train, y_test = load_churn_data()

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

# figure the values of each data point
num = 400
idx = np.arange(1, num + 1)

w = LRPCA.coef_
b = LRPCA.intercept_
dist = np.array([abs(np.dot(w, X_train[i]) + b) for i in range(num)])
dist /= np.linalg.norm(w)

s0 = plt.scatter(idx[y_train[:num]==0], dist[y_train[:num]==0], c='r', marker='x')
s1 = plt.scatter(idx[y_train[:num]==1], dist[y_train[:num]==1], c='b', marker='o')
plt.xlabel('data points')
plt.ylabel('distance')
plt.title('Distance of each data point to the hyper-plane')
plt.legend(handles=[s0, s1], labels=['type 0', 'type 1'])
plt.show()

# under the pca data, do compute the SV for all the data points needs about 2 days, 
# so this can only be done on less data points, or turn to data groups
svs = shapley_values(X_train[:num], y_train[:num], X_test, y_test, evaluate='loss')

s0 = plt.scatter(idx[y_train[:num]==0], svs[y_train[:num]==0], c='r', marker='x')
s1 = plt.scatter(idx[y_train[:num]==1], svs[y_train[:num]==1], c='b', marker='o')
plt.xlabel('data points')
plt.ylabel('shapley value')
plt.title('Shapley Values of each data point')
plt.legend(handles=[s0, s1], labels=['type 0', 'type 1'])
plt.show()