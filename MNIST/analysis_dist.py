from tokenize import group
import torch
import torchvision
from torch.utils.data import DataLoader
from time import time
from utils.utils import *
from utils.model import *

def train_(model, k, train_dataset, test_dataset, epochs=10, show=False):
    train_loader = DataLoader(train_dataset, 128)
    test_loader = DataLoader(test_dataset, 500)
    loss_fn = torch.nn.CrossEntropyLoss()
    net = model(k).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    high_acc = 0
    for i in range(epochs):
        net.train()
        train_loss = 0
        for data in train_loader:
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = net(inputs)
            loss = loss_fn(outputs, targets)
            train_loss += loss
            # optimize the network
            optimizer.zero_grad()
            loss.backward() # back propagation
            optimizer.step()
        
        net.eval()
        with torch.no_grad():
            test_loss, test_acc = 0, 0
            for data in test_loader:
                inputs, targets = data
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = net(inputs)
                loss = loss_fn(outputs, targets)
                test_loss += loss
                test_acc += sum(torch.argmax(F.softmax(outputs, dim=1), dim=1) == targets)

        high_acc = max(test_acc, high_acc)

        if show:
            print('epoch {}'.format(i))
            print('\ttrain loss: {}'.format(train_loss))
            print('\ttest loss: {}'.format(test_loss))
            print('\ttest accuracy: {}'.format(test_acc / len(test_dataset)))
    
    print('hightest accuracy: {}'.format(high_acc / len(test_dataset)))

    return net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trans = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(0.1307, 0.3081)
])

train_class_num = 600
test_class_num = 500

train_dataset = MyMNIST(root='MNIST/data', train=True, transform=trans, download=True)
test_dataset = MyMNIST(root='MNIST/data', train=False, transform=trans, download=True)
class_types = [0, 1, 2, 3]
train_dataset.truncate(class_types, train_class_num)
test_dataset.truncate(class_types, test_class_num)
torch.save(train_dataset, 'MNIST/calculation/distance/train.pt')
train_dataset = torch.load('MNIST/calculation/distance/train.pt')
train_loader = DataLoader(train_dataset, 1)
train_num = len(train_dataset)

model = LeNet
print('use all data points:')
net = train_(model, len(class_types), train_dataset, test_dataset, show=False)

# compute dist and delta dist
dist = np.zeros(train_dataset.targets.shape)
i = 0
for img, target in train_loader:
    img = img.to(device)
    target = target.to(device)
    output = net(img)
    predict_target = torch.argmax(F.softmax(output, dim=1), dim=1)
    r, img_pert, lx = deepfool(img.squeeze(0), net, len(class_types), overshoot=1e-5, max_iter=1000)
    # print('{}: true target {}, predict target {}, target after deepfool {}'.format(i, target, predict_target, lx))
    if predict_target != lx:
        dist[i] = np.linalg.norm(r.flatten())
        # print('distance to the border {}'.format(dist[i]))
    else:
        print('failed to change the label')
    i += 1

np.save('MNIST/calculation/distance/dist', dist)
dist = np.load('MNIST/calculation/distance/dist.npy')

delta_dist = np.zeros(train_dataset.targets.shape)
i = 0
for img, target in train_loader:
    start = time()
    img = img.to(device)
    target = target.to(device)

    delta_dataset = torch.load('MNIST/calculation/distance/train.pt')
    delta_dataset.data = torch.vstack((delta_dataset.data[:i], delta_dataset.data[i + 1:]))
    delta_dataset.targets = torch.hstack((delta_dataset.targets[:i], delta_dataset.targets[i + 1:]))
    delta_net = train_(model, len(class_types), delta_dataset, test_dataset)
    output = delta_net(img)
    predict_target = torch.argmax(F.softmax(output, dim=1), dim=1)
    r, img_pert, lx = deepfool(img.squeeze(0), delta_net, len(class_types), overshoot=1e-5, max_iter=1000)
    end = time()
    print('{}: true target {}, predict target {}, target after deepfool {}, time spent {}'.format(i, target, predict_target, lx, end - start))
    if predict_target != lx:
        delta_dist[i] = abs(dist[i] - np.linalg.norm(r.flatten()))
        print('delta distance {}'.format(delta_dist[i]))
    else:
        print('failed to change the label')
    i += 1

np.save('MNIST/calculation/distance/delta_dist', delta_dist)
delta_dist = np.load('MNIST/calculation/distance/delta_dist.npy')

# check out impact of dist and delta dist
groups = [4, 5, 6]
for g in groups:
    print('================ devide into {} groups ================'.format(g))
    # distance:
    rank_dist = np.argsort(dist)

    dist_idxs = []
    for i in range(g):
        dist_idx = rank_dist[train_num * i // g: train_num * (i + 1) // g].copy()
        dist_idxs.append(dist_idx)

    for i, idx in enumerate(dist_idxs):
        dataset = torch.load('MNIST/calculation/distance/train.pt')
        dataset.data =  dataset.data[idx]
        dataset.targets = dataset.targets[idx]
        print('\nuse dist{} data points:'.format(i))
        train_(model, len(class_types), dataset, test_dataset, show=False)

    # delta distance:
    rank_delta_dist = np.argsort(delta_dist)

    delta_dist_idxs = []
    for i in range(g):
        delta_dist_idx = rank_delta_dist[train_num * i // g: train_num * (i + 1) // g].copy()
        delta_dist_idxs.append(delta_dist_idx)

    for i, idx in enumerate(delta_dist_idxs):
        dataset = torch.load('MNIST/calculation/distance/train.pt')
        dataset.data =  dataset.data[idx]
        dataset.targets = dataset.targets[idx]
        print('\nuse delta_dist{} data points:'.format(i))
        train_(model, len(class_types), dataset, test_dataset, show=False)

    # check the relation between distance and delta distance
    for d in dist_idxs:
        for dd in delta_dist_idxs:
            print(sum(np.in1d(d, dd)), end=' ')
        print()