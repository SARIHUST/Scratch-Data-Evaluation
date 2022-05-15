'''
    This script computes the TMC-Shapley Values of part of the MNIST training set data
points and the distance of the data points to the decision border of the network by adding
perturbation to the original image until the network classifies it to a different label.
'''

import torch
import torch.nn.functional as F
import torchvision
from utils.utils import *
from utils.model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trans = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(0.1307, 0.3081)
])

train_dataset = MyMNIST(root='MNIST/data', train=True, transform=trans, download=True)
test_dataset = MyMNIST(root='MNIST/data', train=False, transform=trans, download=True)
class_types = [0, 1, 2]
train_dataset.truncate(class_types, 600)
test_dataset.truncate(class_types, 300)

torch.save(train_dataset.data, 'MNIST/calculation/train_data.pt')
torch.save(test_dataset.data, 'MNIST/calculation/test_data.pt')
torch.save(train_dataset.targets, 'MNIST/calculation/train_targets.pt')
torch.save(test_dataset.targets, 'MNIST/calculation/test_targets.pt')

model = LeNet
sv, net, sv_it = shapley_values(model, train_dataset, test_dataset, k=len(class_types))

print(sv)
np.save('MNIST/calculation/shapley-values', sv)
np.save('MNIST/calculation/sv_it', sv_it)

dist = np.zeros(train_dataset.targets.shape)
real_labels = np.zeros(train_dataset.targets.shape)
predict_labels = np.zeros(train_dataset.targets.shape)
deepfool_labels = np.zeros(train_dataset.targets.shape)
net.eval()
outputs = predict(net, test_dataset)
y_predict = torch.argmax(F.softmax(outputs, dim=0), 1).detach().cpu().numpy()
print('total accuracy: {}'.format(accuracy_score(test_dataset.targets, y_predict)))

for i in range(len(train_dataset)):
    img = torchvision.transforms.Normalize(0.1307, 0.3081)(train_dataset.data[i].unsqueeze(0).float())
    img_c = img.unsqueeze(0).to(device)
    target = train_dataset.targets[i]
    predict_target = torch.argmax(net(img_c), 1)
    r, img_pert, lx = deepfool(img, net, 3, overshoot=50000, max_iter=1000)
    print('{}: true target {}, predict target {},target after deepfool {}'.format(i, target, predict_target, lx))
    if predict_target != lx:
        dist[i] = np.linalg.norm(r.flatten())
        print('distance to the border {}'.format(dist[i]))
    else:
        print('failed to change the label')
    real_labels[i], predict_labels[i], deepfool_labels[i] = target, predict_target, lx
distance = np.vstack((real_labels, predict_labels, deepfool_labels, dist))
np.save('MNIST/calculation/distance', distance)