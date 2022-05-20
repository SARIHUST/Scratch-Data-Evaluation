from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.datasets import MNIST
from zmq import device
from model import *
from sklearn.metrics import accuracy_score

seed = 423
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TempDataset(Dataset):
    def __init__(self, data, targets) -> None:
        super().__init__()
        self.data = data
        self.targets = targets
        
    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)

def deepfool(img, net, k=10, overshoot=0.02, max_iter=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img.to(device)
    net.to(device)
    p_img = net(img.unsqueeze(0))
    idx = np.array(F.softmax(p_img.flatten(), dim=0).detach().numpy()).argsort()[::-1][0:k]

    label = idx[0]
    img_pert = img.clone().detach()
    w, r = np.zeros(img.shape), np.zeros(img.shape)
    
    x = img_pert.unsqueeze(0).float()
    x.requires_grad = True
    px = net(x)[0]
    lx = label
    it = 0
    while lx == label and it < max_iter:
        pert = np.inf
        px[idx[0]].backward(retain_graph=True)
        orig_grad = x.grad.data.numpy().copy()

        # find the minimum perturbation to change to another label
        for i in range(1, k):
            x.grad = None
            px[idx[i]].backward(retain_graph=True)
            i_grad = x.grad.data.numpy().copy()

            w_i = i_grad - orig_grad
            f_i = px[idx[i]] - px[idx[0]]
            pert_i = abs(f_i) / np.linalg.norm(w_i.flatten())

            if pert_i < pert:
                pert = pert_i
                w = w_i
        # compute the total perturbation at this round
        pert = pert.detach().numpy()
        rx = (pert + 1e-7) * w / np.linalg.norm(w)
        r += rx.squeeze(0)
        # compute the new image
        img_pert = (img + (1 + overshoot) * torch.from_numpy(r)).to(device)
        x = img_pert.unsqueeze(0).float()
        x.requires_grad = True
        px = net(x)[0]
        # print(px)
        lx = torch.argmax(px)

        it += 1

    r = (1 + overshoot) * r
    return r, img_pert, lx


class MyMNIST(MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=True):
        super().__init__(root, train, transform, target_transform, download)

    def truncate(self, classes: list, type_size=None):
        '''
        parameters 
            classes: a list that contains the different types you want
            type_size: the number of data points you want for each type
        '''
        idx = list(range(len(self.targets)))
        trunc_idx = []
        for ci in classes:
            ci_idx = [i for i in idx if self.targets[i] == ci]
            np.random.shuffle(ci_idx)
            if type_size is None:
                type_size_i = sum(self.targets == ci)
            else:
                type_size_i = type_size
            trunci = ci_idx[:type_size_i]
            trunc_idx.extend(trunci)
        np.random.shuffle(trunc_idx)
        self.data = self.data[trunc_idx]
        self.targets = self.targets[trunc_idx]

def train(model, k, train_dataset, nan=False):
    train_loader = DataLoader(train_dataset, 128)
    epochs = 4
    loss_fn = torch.nn.CrossEntropyLoss()
    net = model(k).to(device)
    # optimizer = torch.optim.SGD(net.parameters(), lr=7e-3, momentum=0.5)
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-5)
    for i in range(epochs):
        # print('epoch {} starts'.format(i))
        net.train()
        total_train_accurate = 0
        for data in train_loader:
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            if len(inputs.shape) == 3:
                inputs = inputs.unsqueeze(1)
            outputs = net(inputs)
            loss = loss_fn(outputs, targets)
            accurate = sum(torch.argmax(outputs, 1) == targets)
            total_train_accurate += accurate
            # optimize the network
            optimizer.zero_grad()
            loss.backward() # back propagation
            optimizer.step()
        if nan:
            print('total tarin accuracy: {}'.format(total_train_accurate / len(train_dataset)))
    return net

def predict(net, test_dataset, nan=False):
    test_loader = DataLoader(test_dataset, len(test_dataset))
    if nan:
        test_loader = DataLoader(test_dataset, 1)
    net.eval()
    for data in test_loader:
        inputs, _ = data
        inputs = inputs.to(device)
        outputs = net(inputs)
    return outputs

def shapley_values(model, train_dataset, test_dataset, k=10, max_p=2):
    '''
        The function is implemented based on the TMC-SV algorithm
    '''
    trans = torchvision.transforms.Normalize(0.1307, 0.3081)
    n = len(train_dataset)
    phais = np.zeros(n)
    t = 0
    fnet = train(model, k, train_dataset, True)
    outputs = predict(fnet, test_dataset)
    outputs_orig = torch.randn(outputs.shape)
    # use loss as score
    loss_fn = nn.CrossEntropyLoss()
    total_score = -loss_fn(outputs, test_dataset.targets)
    orig_score = -loss_fn(outputs_orig, test_dataset.targets)

    epsdata = trans(train_dataset.data.float()[:int(n * 0.9)])
    epstarget = train_dataset.targets[:int(n * 0.9)]
    epsdataset = TempDataset(epsdata, epstarget)
    epsnet = train(model, k, epsdataset, True)
    outputs = predict(epsnet, test_dataset)
    eps_score = -loss_fn(outputs, test_dataset.targets)

    # use accuracy as score
    # orig_predict = torch.argmax(F.softmax(outputs_orig, dim=1), dim=1).detach().cpu().numpy()
    y_predict = torch.argmax(F.softmax(outputs, dim=1), dim=1).detach().cpu().numpy()
    # total_score = accuracy_score(test_dataset.targets, y_predict)
    # orig_score = accuracy_score(test_dataset.targets, orig_predict)

    print('total accuracy: {}'.format(accuracy_score(test_dataset.targets, y_predict)))
    print('total loss: {}'.format(total_score))
    epsilon = float(abs(total_score - eps_score)) / 2
    print('epsilon: {}'.format(epsilon))
    record = [[0] for _ in range(n)]

    while t < max_p * n:
        old_phais = phais.copy()
        start = time()
        t += 1
        vs = np.zeros(n + 1)
        vs[0] = orig_score
        pai_t = np.random.permutation(np.arange(0, n, step=1))
        for j in range(1, n + 1):
            if j % 100 == 0:
                print(j)
            idx = pai_t[j - 1]
            if total_score - vs[j - 1] <= epsilon:
                vs[j] = vs[j - 1]
            else:
                data = trans(train_dataset.data.float()[pai_t[:j]])
                label = train_dataset.targets[pai_t[:j]]
                dataset = TempDataset(data, label)
                net = train(model, k, dataset)
                outputs = predict(net, test_dataset)
                while outputs[0][0] != outputs[0][0]:
                    print(j)
                    print('NaN problem')
                    train(model, k, dataset, True)
                    outputs = predict(net, test_dataset, True)
                vs[j] = -loss_fn(outputs, test_dataset.targets)
                # y_predict = torch.argmax(F.softmax(outputs, dim=1), 1).detach().cpu().numpy()
                # vs[j] = accuracy_score(test_dataset.targets, y_predict)
            phais[idx] = phais[idx] * (t - 1) / t + (vs[j] - vs[j - 1]) / t
        end = time()
        print('iteration {}, converge {}(remains zero {}), time cost {}'.format(t, sum(abs(old_phais - phais) < epsilon / 30), sum(phais == 0), end - start))
        for ik in range(n):
            record[ik].append(phais[ik])
        if sum(abs(old_phais - phais) < epsilon / 30) - sum(phais == 0) == n:
            break

    return phais, fnet, record


if __name__ == '__main__':
    trans = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(0.1307, 0.3081)
    ])

    train_dataset = MyMNIST('MNIST/data', train=True, transform=trans, download=True)
    train_dataset.truncate([0, 1, 2], 500)
    test_dataset = MyMNIST('MNIST/data', train=False, transform=trans, download=True)
    test_dataset.truncate([0, 1, 2], 100)
    model = LeNet
    
    res = shapley_values(model, train_dataset, test_dataset)

    net = model(classes=3)
    net = train(net, train_dataset)
    train_dataset.data = torchvision.transforms.Normalize(0.1307, 0.3081)(train_dataset.data.float())
    net.eval()
    for i in range(len(train_dataset)):
        img = train_dataset.data[i].unsqueeze(0)
        target = train_dataset.targets[i]
        predict_target = torch.argmax(net(img.unsqueeze(0)), 1)
        r, img_pert, lx = deepfool(img, net, 3, overshoot=500, max_iter=1000)
        print('true target {}, predict target {},target after deepfool {}'.format(target, predict_target, lx))
        print(np.linalg.norm(r.flatten()))
