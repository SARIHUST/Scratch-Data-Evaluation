'''
    This script shows the SV vs Distance of the data points.
'''
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from utils.utils import *
from utils.model import LeNet
from sklearn.metrics import accuracy_score

sv = np.load('MNIST/calculation/sv_400.npy')
dist = np.load('MNIST/calculation/distance_400.npy')
# dist -> real_labels, predict_labels, deepfool_labels, distance
show_num = 100
calc_num = len(sv)

loss_fn = nn.CrossEntropyLoss()

train_dataset = torch.load('MNIST/calculation/train.pt')
test_dataset = torch.load('MNIST/calculation/test.pt')

# consider -- benchmark
rank_sv = np.argsort(sv)[::-1]
high_sv_idx = rank_sv[:calc_num // 2].copy()
low_sv_idx = rank_sv[calc_num // 2:].copy()

total_net = train(LeNet, 2, train_dataset)
total_outputs = predict(total_net, test_dataset)
total_predict = torch.argmax(F.softmax(total_outputs, dim=1), dim=1).detach().cpu().numpy()
total_loss = loss_fn(total_outputs, test_dataset.targets)
total_acc = accuracy_score(test_dataset.targets, total_predict)
print('total accuracy: {}, loss: {}'.format(total_acc, total_loss))

high_sv_dataset = torch.load('MNIST/calculation/train.pt')
high_sv_dataset.data = high_sv_dataset.data[high_sv_idx]
high_sv_dataset.targets = high_sv_dataset.targets[high_sv_idx]
high_net = train(LeNet, 2, high_sv_dataset)
high_outputs = predict(high_net, test_dataset)
high_predict = torch.argmax(F.softmax(high_outputs, dim=1), dim=1).detach().cpu().numpy()
high_loss = loss_fn(high_outputs, test_dataset.targets)
high_acc = accuracy_score(test_dataset.targets, high_predict)
print('high accuracy: {}, loss: {}'.format(high_acc, high_loss))

low_sv_dataset = torch.load('MNIST/calculation/train.pt')
low_sv_dataset.data = low_sv_dataset.data[low_sv_idx]
low_sv_dataset.targets = low_sv_dataset.targets[low_sv_idx]
low_net = train(LeNet, 2, low_sv_dataset)
low_outputs = predict(low_net, test_dataset)
low_predict = torch.argmax(F.softmax(low_outputs, dim=1), dim=1).detach().cpu().numpy()
low_loss = loss_fn(low_outputs, test_dataset.targets)
low_acc = accuracy_score(test_dataset.targets, low_predict)
print('low accuracy: {}, loss: {}'.format(low_acc, low_loss))

# consider distance
distance = dist[3]
rank_distance = np.argsort(distance)
near_dist_idx = rank_distance[:calc_num // 2].copy()
far_dist_idx = rank_distance[calc_num // 2:].copy()

near_dist_dataset = torch.load('MNIST/calculation/train.pt')
near_dist_dataset.data = near_dist_dataset.data[near_dist_idx]
near_dist_dataset.targets = near_dist_dataset.targets[near_dist_idx]
near_net = train(LeNet, 2, near_dist_dataset)
near_outputs = predict(near_net, test_dataset)
near_predict = torch.argmax(F.softmax(near_outputs, dim=1), dim=1).detach().cpu().numpy()
near_loss = loss_fn(near_outputs, test_dataset.targets)
near_acc = accuracy_score(test_dataset.targets, near_predict)
print('near accuracy: {}, loss: {}'.format(near_acc, near_loss))

far_dist_dataset = torch.load('MNIST/calculation/train.pt')
far_dist_dataset.data = far_dist_dataset.data[far_dist_idx]
far_dist_dataset.targets = far_dist_dataset.targets[far_dist_idx]
far_net = train(LeNet, 2, far_dist_dataset)
far_outputs = predict(far_net, test_dataset)
far_predict = torch.argmax(F.softmax(far_outputs, dim=1), dim=1).detach().cpu().numpy()
far_loss = loss_fn(far_outputs, test_dataset.targets)
far_acc = accuracy_score(test_dataset.targets, far_predict)
print('far accuracy: {}, loss: {}'.format(far_acc, far_loss))

# show difference between the 2 methods
sv_show = sv[:show_num]
dist_show = dist[:, :show_num]   

sv_show = np.array([sv_show[i] for i in range(show_num) if dist_show[1][i] != dist_show[2][i]])
dist_dist = np.array([dist_show[3][i] for i in range(show_num) if dist_show[1][i] != dist_show[2][i]])
sv_show = sv_show / sum(sv_show) * 40 + 4
dist_dist = dist_dist / sum(dist_dist) * 100

idx = np.arange(1, show_num + 1)
s1 = plt.scatter(idx[dist_show[1] == dist_show[0]], sv_show[dist_show[1] == dist_show[0]], c='r', marker='x')
s2 = plt.scatter(idx[dist_show[1] != dist_show[0]], sv_show[dist_show[1] != dist_show[0]], c='b', marker='x')
plt.plot(idx, sv_show)
s3 = plt.scatter(idx[dist_show[1] == dist_show[0]], dist_dist[dist_show[1] == dist_show[0]], c='r', marker='o')
s4 = plt.scatter(idx[dist_show[1] != dist_show[0]], dist_dist[dist_show[1] != dist_show[0]], c='b', marker='o')
plt.plot(idx, dist_dist)
plt.xlabel('data points')
plt.ylabel('proportion to total')
plt.title('Shapley Value vs Distance to border')
plt.legend(handles=[s1, s2, s3, s4], labels=['sv-c', 'sv-w', 'dist-c', 'dist-w'])
plt.show()