'''
    This script trains the network used to evaluate the training data points.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from utils.model import LeNet, TryNet
from utils.utils import *
from torch.utils.tensorboard import SummaryWriter
from time import time

# prepare dataset:
trans = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(0.1307, 0.3081)
])

train_dataset = MyMNIST(root='MNIST/data', train=True, transform=trans, download=True)
test_dataset = MyMNIST(root='MNIST/data', train=False, transform=trans, download=True)
train_size = len(train_dataset)
test_size = len(test_dataset)

train_loader = DataLoader(train_dataset, 128)
test_loader = DataLoader(test_dataset, 1000)

# create network, set environment
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = LeNet().to(device)
loss_fn = nn.CrossEntropyLoss().to(device)

learning_rate = 1e-3
# optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, momentum=0.5)

epochs = 10
train_rounds = 0

writer = SummaryWriter('MNIST/log')

# train and test:
for i in range(epochs):
    print('-------epoch {}-------'.format(i))
    start_time = time()
    
    net.train()
    total_train_accurate = 0
    for data in train_loader:
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = net(inputs)
        loss = loss_fn(outputs, targets)
        accurate = sum(torch.argmax(outputs, 1) == targets)
        total_train_accurate += accurate

        # optimize the network
        optimizer.zero_grad()
        loss.backward() # back propagation
        optimizer.step()

        if train_rounds % 100 == 0:
            end_time = time()
            print('train round {}, Loss: {}, time spent: {}'.format(train_rounds, loss.item(), end_time - start_time))
            writer.add_scalar('train_loss on rounds(batches of 128)', loss, train_rounds)
        train_rounds += 1

    print('epoch {} total train accuracy: {}'.format(i, total_train_accurate / train_size))
    writer.add_scalar('train_accuracy on epochs', total_train_accurate / train_size, i)

    net.eval()
    total_test_loss = 0
    total_test_accurate = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = net(inputs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss
            accurate = sum(torch.argmax(outputs, 1) == targets)
            total_test_accurate += accurate
    print('epoch {} total test loss: {}'.format(i, total_test_loss))
    print('epoch {} total test accuracy: {}'.format(i, total_test_accurate / test_size))
    writer.add_scalar('test_loss on epochs', total_test_loss, i)
    writer.add_scalar('test_accuracy on epoch', total_test_accurate / test_size, i)

    torch.save(net, 'MNIST/model/LeNet_{}.pth'.format(i))