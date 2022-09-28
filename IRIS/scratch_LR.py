'''
    This script is to test if the distance of a data point to the decision boundary of a
    Logistic Regression classfier is related to the value of the data point by using the 
    mini-dataset sklearn.datasets.iris
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from utils import shapley_values

np.random.seed(110810423)

iris = datasets.load_iris()

X = iris.data[iris.target < 2]
y = iris.target[iris.target < 2]

# show the data points of the first 2 types on the first 2 features
X = X[y < 2, :2]
y = y[y < 2]
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='b', marker='o')
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='r', marker='x')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()

# train the LR model
X_train, X_test, y_train, y_test = train_test_split(X, y)
LRClassifier = LogisticRegression()
LRClassifier.fit(X_train, y_train)
print(LRClassifier.coef_)
print(LRClassifier.intercept_)
y_predict = LRClassifier.predict(X_test)
total_accuracy = accuracy_score(y_pred=y_predict, y_true=y_test)
total_loss = log_loss(y_test, y_predict)
print('accuracy score: {}'.format(total_accuracy))
print('loss score: {}'.format(total_loss))

# plot the decision boundary of the LR model
w = np.array(LRClassifier.coef_)
x1 = np.arange(4, 8, step=0.1)
x2 = -(w[0][0] * x1 + LRClassifier.intercept_[0]) / w[0][1]
plt.plot(x1, x2)
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='b', marker='o')
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='r', marker='x')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('Decision Boundary')
plt.show()

exit()

# Now we try to find out the most valuable data points according to LOO strategy, Shapley Values
# Leave One Out Strategy
loo_loss = np.zeros(len(X_train))
loo_values = np.zeros(len(X_train))
for i in range(len(X_train)):
    X_ = np.vstack((X_train[:i], X_train[i + 1:]))
    y_ = np.hstack((y_train[:i], y_train[i + 1:]))
    LR = LogisticRegression()
    LR.fit(X_, y_)
    y_predict = LR.predict(X_test)
    loss = log_loss(y_pred=y_predict, y_true=y_test)
    loo_loss[i] = loss
    loo_values[i] = loss - total_loss   
    # assume that a data point should have positive value, then the loss should come down

print(loo_loss)
print(loo_values)   # every data point's value is 0.0 under this method

# Shapley Values with Monte-Carlo methods
# svs, sv_it = shapley_values(X_train, y_train, X_test, y_test, max_p=150)
# np.save('IRIS/svs', svs)
# np.save('IRIS/sv_it', sv_it)

svs = np.load('IRIS/svs.npy')   # use the saved data to jump pass shapley values computing
sv_it = np.load('IRIS/sv_it.npy')

idx = np.arange(1, len(X_train) + 1)
s0 = plt.scatter(idx[y_train==0], svs[y_train==0], c='r', marker='x')
s1 = plt.scatter(idx[y_train==1], svs[y_train==1], c='b', marker='o')
plt.xlabel('data points')
plt.ylabel('shapley value')
plt.title('Shapley Values of each data point')
plt.legend(handles=[s0, s1], labels=['type 0', 'type 1'])
plt.show()

# show the change of shapley value against the number of iterations
idx = np.arange(1, len(sv_it[0]) + 1)
show_cases = [4, 23, 45, 67]

plt.subplot(2, 2, 1)
plt.plot(idx, sv_it[show_cases[0]])
plt.xlabel('iterations')
plt.ylabel('shapley value({}th)'.format(show_cases[0]))
plt.subplot(2, 2, 2)
plt.plot(idx, sv_it[show_cases[1]])
plt.xlabel('iterations')
plt.ylabel('shapley value({}th)'.format(show_cases[1]))
plt.subplot(2, 2, 3)
plt.plot(idx, sv_it[show_cases[2]])
plt.xlabel('iterations')
plt.ylabel('shapley value({}th)'.format(show_cases[2]))
plt.subplot(2, 2, 4)
plt.plot(idx, sv_it[show_cases[3]])
plt.xlabel('iterations')
plt.ylabel('shapley value({}th)'.format(show_cases[3]))
plt.show()

# Compute the distance of the training data points to the decision boundary
w = LRClassifier.coef_
b = LRClassifier.intercept_

dist = np.array([abs(np.dot(w, x) + b).item() for x in X_train])
dist /= np.linalg.norm(w)
print(dist)
idx = np.arange(1, len(dist) + 1)
s0 = plt.scatter(idx[y_train==0], dist[y_train==0], c='r', marker='x')
s1 = plt.scatter(idx[y_train==1], dist[y_train==1], c='b', marker='o')
plt.xlabel('data points')
plt.ylabel('distance')
plt.title('Distance of each data point to the hyper-plane')
plt.legend(handles=[s0, s1], labels=['type 0', 'type 1'])
plt.show()

# compare the 2 methods

svs_ = svs / sum(svs) * 250
dist_ = dist / sum(dist) * 100
s0 = plt.scatter(idx, svs_, c='r', marker='x')
plt.plot(idx, svs_)
s1 = plt.scatter(idx, dist_, c='b', marker='o')
plt.plot(idx, dist_)
plt.xlabel('data points')
plt.ylabel('proportion\' to total')
plt.title('shapley value vs distance to border')
plt.legend(handles=[s0, s1], labels=['sv', 'dist'])
plt.show()

# Consider the impact of a data point on the decision boundary when removed from training
# LOO distance change:
loo_dist = np.zeros(len(X_train))
for i in range(len(X_train)):
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

print(loo_dist)

s0 = plt.scatter(idx, loo_dist, c='r', marker='x')
plt.xlabel('data points')
plt.ylabel('delta distance')
plt.title('Delta distance by LOO strategy')
plt.show()

loo_dist_ = loo_dist / sum(loo_dist) * 30
s0 = plt.scatter(idx, svs_, c='r', marker='x')
plt.plot(idx, svs_)
s1 = plt.scatter(idx, loo_dist_, c='b', marker='o')
plt.plot(idx, loo_dist_)
plt.xlabel('data points')
plt.ylabel('proportion\' to total')
plt.title('shapley value vs delta distance')
plt.legend(handles=[s0, s1], labels=['sv', 'delta dist'])
plt.show()

dist_ = dist / sum(dist) * 25 + 0.25
loo_dist_ = loo_dist / sum(loo_dist) * 5
s0 = plt.scatter(idx, dist_, c='r', marker='x')
plt.plot(idx, dist_)
s1 = plt.scatter(idx, loo_dist_, c='b', marker='o')
plt.plot(idx, loo_dist_)
plt.xlabel('data points')
plt.ylabel('proportion\' to total')
plt.title('distance vs delta distance')
plt.legend(handles=[s0, s1], labels=['dist', 'delta dist'])
plt.show()
# This figure shows that the data points that are closer to the decision boundary has an impact
# on the boundary that is larger than those that are further. However when considered with the 
# shapley value, no relationship can be seen directly.

# compute the pearson correlation coefficient
mat = np.vstack((svs, dist, loo_dist))
print(mat)
rho = np.corrcoef(mat)
print(rho)

cov = np.cov(mat)
print(cov)