'''
    This script is to test the decision boundary of Logistic Regression
    by using the dataset sklearn.datasets.iris
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
# X = X[y < 2, :2]    # turn it to a 2d problem
# y = y[y < 2]

# show the data points of the first 2 types on the first 2 features
# plt.scatter(X[y == 0, 0], X[y == 0, 1], c='b', marker='o')
# plt.scatter(X[y == 1, 0], X[y == 1, 1], c='r', marker='x')
# plt.xlabel(iris.feature_names[0])
# plt.ylabel(iris.feature_names[1])
# plt.show()

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
# w = np.array(LRClassifier.coef_)
# x1 = np.arange(4, 8, step=0.1)
# x2 = -(w[0][0] * x1 + LRClassifier.intercept_[0]) / w[0][1]
# plt.plot(x1, x2)
# plt.scatter(X[y == 0, 0], X[y == 0, 1], c='b', marker='o')
# plt.scatter(X[y == 1, 0], X[y == 1, 1], c='r', marker='x')
# plt.xlabel(iris.feature_names[0])
# plt.ylabel(iris.feature_names[1])
# plt.title('Decision Boundary')
# plt.show()

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
print(loo_values)

# Shapley Values with Monte-Carlo methods
svs, rs = shapley_values(X_train, y_train, X_test, y_test)
print(svs)

idx = np.arange(1, len(X_train) + 1)
# plt.scatter(idx, loo_values, c='b', marker='o')
s0 = plt.scatter(idx[y_train==0], svs[y_train==0], c='r', marker='x')
s1 = plt.scatter(idx[y_train==1], svs[y_train==1], c='b', marker='o')
plt.xlabel('data points')
plt.ylabel('shapley value')
plt.title('Shapley Values of each data point')
plt.legend(handles=[s0, s1], labels=['type 0', 'type 1'])
plt.show()

idx = np.arange(1, len(rs) + 1)
plt.plot(idx, rs)
plt.show()
exit()

# Compute the distance of the training data points to the decision boundary
w = LRClassifier.coef_
b = LRClassifier.intercept_

dist = np.array([abs(np.dot(w, x) + b) for x in X_train])
dist /= np.linalg.norm(w)
print(dist)
s0 = plt.scatter(idx[y_train==0], dist[y_train==0], c='r', marker='x')
s1 = plt.scatter(idx[y_train==1], dist[y_train==1], c='b', marker='o')
plt.xlabel('data points')
plt.ylabel('distance')
plt.title('Distance of each data point to the hyper-plane')
plt.legend(handles=[s0, s1], labels=['type 0', 'type 1'])
plt.show()