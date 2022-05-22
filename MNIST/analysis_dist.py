import torch
import torchvision
from torch.utils.data import DataLoader
from utils.utils import *
from utils.model import *

def train_(model, k, train_dataset, test_dataset, show=False):
    train_loader = DataLoader(train_dataset, 128)
    test_loader = DataLoader(test_dataset, 500)
    epochs = 6
    loss_fn = torch.nn.CrossEntropyLoss()
    net = model(k).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
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

        if show:
            print('epoch {}'.format(i))
            print('\ttrain loss: {}'.format(train_loss))
            print('\ttest loss: {}'.format(test_loss))
            print('\ttest accuracy: {}'.format(test_acc / len(test_dataset)))
    
    return net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weightPath = 'MNIST/model/distnet_weight.pth'

trans = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(0.1307, 0.3081)
])

train_num = 1000

# train_dataset = MyMNIST(root='MNIST/data', train=True, transform=trans, download=True)
# test_dataset = MyMNIST(root='MNIST/data', train=False, transform=trans, download=True)
class_types = [0, 1, 2, 3, 4]
# train_dataset.truncate(class_types, train_num)
# test_dataset.truncate(class_types, 500)
# torch.save(train_dataset, 'MNIST/calculation/distance/train.pt')
# torch.save(test_dataset, 'MNIST/calculation/distance/test.pt')
train_dataset = torch.load('MNIST/calculation/distance/train.pt')
test_dataset = torch.load('MNIST/calculation/distance/test.pt')

model = LeNet
print('use all data points:')
net = train_(model, len(class_types), train_dataset, test_dataset, show=True)

# compute dist and delta dist
# dist = np.zeros(train_dataset.targets.shape)
# failure = 0
# for i in range(len(train_dataset)):
#     img = torchvision.transforms.Normalize(0.1307, 0.3081)(train_dataset.data[i].unsqueeze(0).float())
#     img_c = img.unsqueeze(0).to(device)
#     target = train_dataset.targets[i]
#     output = net(img_c)
#     predict_target = torch.argmax(F.softmax(output, dim=1), dim=1)
#     r, img_pert, lx = deepfool(img, net, len(class_types), overshoot=1e-10, max_iter=1000)
#     print('{}: true target {}, predict target {}, target after deepfool {}'.format(i, target, predict_target, lx))
#     if predict_target != lx:
#         dist[i] = np.linalg.norm(r.flatten())
#         print('distance to the border {}'.format(dist[i]))
#     else:
#         print('failed to change the label')
#         failure += 1
# np.save('MNIST/calculation/distance/dist_{}'.format(train_num), dist)
# print('deepfool failed on {} data points'.format(failure))

# delta_dist = np.zeros(train_dataset.targets.shape)
# failure = 0
# for i in range(len(train_dataset)):
#     img = torchvision.transforms.Normalize(0.1307, 0.3081)(train_dataset.data[i].unsqueeze(0).float())
#     img_c = img.unsqueeze(0).to(device)
#     target = train_dataset.targets[i]

#     delta_train_test = torch.load('MNIST/calculation/distance/train.pt')
#     delta_net = train_(model, len(class_types), delta_train_test, test_dataset)
#     output = delta_net(img_c)
#     predict_target = torch.argmax(F.softmax(output, dim=1), dim=1)
#     r, img_pert, lx = deepfool(img, delta_net, len(class_types), overshoot=1e-10, max_iter=1000)
#     print('{}: true target {}, predict target {}, target after deepfool {}'.format(i, target, predict_target, lx))
#     if predict_target != lx:
#         delta_dist[i] = abs(dist[i] - np.linalg.norm(r.flatten()))
#         print('distance to the border {}'.format(delta_dist[i]))
#     else:
#         print('failed to change the label')
#         failure += 0
# np.save('MNIST/calculation/distance/delta_dist_{}'.format(train_num), delta_dist)
# print('deepfool failed on {} data points'.format(failure))

# check out impact of dist and delta dist
# distance:
dist = np.load('MNIST/calculation/distance/dist_{}.npy'.format(train_num))
rank_distance = np.argsort(dist)

dist1_idx = rank_distance[:train_num // 3].copy()
dist2_idx = rank_distance[train_num // 3: 2 * train_num // 3].copy()
dist3_idx = rank_distance[2 * train_num // 3:].copy()

dist1_dataset = torch.load('MNIST/calculation/distance/train.pt')
dist1_dataset.data = dist1_dataset.data[dist1_idx]
dist1_dataset.targets = dist1_dataset.targets[dist1_idx]

dist2_dataset = torch.load('MNIST/calculation/distance/train.pt')
dist2_dataset.data = dist2_dataset.data[dist2_idx]
dist2_dataset.targets = dist2_dataset.targets[dist2_idx]

dist3_dataset = torch.load('MNIST/calculation/distance/train.pt')
dist3_dataset.data = dist3_dataset.data[dist3_idx]
dist3_dataset.targets = dist3_dataset.targets[dist3_idx]

print('\n\nuse dist1 data points:')
dist1_net = train_(model, len(class_types), dist1_dataset, test_dataset, show=True)

print('\n\nuse dist2 data points:')
dist2_net = train_(model, len(class_types), dist2_dataset, test_dataset, show=True)

print('\n\nuse dist3 data points:')
dist3_net = train_(model, len(class_types), dist3_dataset, test_dataset, show=True)

# delta distance:
delta_dist = np.load('MNIST/calculation/distance/dist_{}.npy'.format(train_num))
rank_delta_dist = np.argsort(delta_dist)

delta_dist1_idx = rank_delta_dist[:train_num // 3].copy()
delta_dist2_idx = rank_delta_dist[train_num // 3: 2 * train_num // 3].copy()
delta_dist3_idx = rank_delta_dist[2 * train_num // 3:].copy()

delta_dist1_dataset = torch.load('MNIST/calculation/distance/train.pt')
delta_dist1_dataset.data = delta_dist1_dataset.data[delta_dist1_idx]
delta_dist1_dataset.targets = delta_dist1_dataset.targets[delta_dist1_idx]

delta_dist2_dataset = torch.load('MNIST/calculation/distance/train.pt')
delta_dist2_dataset.data = delta_dist2_dataset.data[delta_dist2_idx]
delta_dist2_dataset.targets = delta_dist2_dataset.targets[delta_dist2_idx]

delta_dist3_dataset = torch.load('MNIST/calculation/distance/train.pt')
delta_dist3_dataset.data = delta_dist3_dataset.data[delta_dist3_idx]
delta_dist3_dataset.targets = delta_dist3_dataset.targets[delta_dist3_idx]

print('\n\nuse delta_dist1 data points:')
delta_dist1_net = train_(model, len(class_types), delta_dist1_dataset, test_dataset, show=True)

print('\n\nuse delta_dist2 data points:')
delta_dist2_net = train_(model, len(class_types), delta_dist2_dataset, test_dataset, show=True)

print('\n\nuse delta_dist3 data points:')
delta_dist3_net = train_(model, len(class_types), delta_dist3_dataset, test_dataset, show=True)