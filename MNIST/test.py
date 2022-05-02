import torch
import torchvision
from utils.utils import *
from torch.utils.data import DataLoader
import cv2
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = torch.load('MNIST/model/LeNet_9.pth', map_location=device)

trans = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(0.1307, 0.3081)
])

# test on the first image of MNIST-TEST
test_dataset = MyMNIST('MNIST/data', train=False, transform=trans, download=True)

test_loader = DataLoader(test_dataset, 10000)

total_accurate = 0
net.eval()
for data in test_loader:
    inputs, targets = data
    outputs = net(inputs)
    accurate = sum(torch.argmax(outputs, 1) == targets)
    total_accurate += accurate
    
print('accuracy: {}'.format(total_accurate / len(test_dataset)))

# image test on a hand written number 2
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((28, 28)),
    torchvision.transforms.Normalize(0.1307, 0.3081)
])

img = cv2.imread('MNIST/test/t2.png')
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img = transform(img)
img = img.unsqueeze(0)
img = img.to(device)
with torch.no_grad():
    output = net(img)
print('predicts t2 to be {}'.format(torch.argmax(output)))

# test deepfool on t7.jpg
img = cv2.imread('MNIST/test/t7.png')
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img = transform(img).to(device)
r, img_pert, lx = deepfool(img, net, 10, overshoot=100, max_iter=1000)
print(lx)
print(np.linalg.norm(r.flatten()))

exit()

mean = [0.1307]
std = [0.3081]

def clip_tensor(A, minv, maxv):
    A = torch.max(A, minv*torch.ones(A.shape))
    A = torch.min(A, maxv*torch.ones(A.shape))
    return A

clip = lambda x: clip_tensor(x, 0, 255)

tf = torchvision.transforms.Compose([
    torchvision.transforms.Normalize(mean=[0], std=list(map(lambda x: 1 / x, std))),
    torchvision.transforms.Normalize(mean=list(map(lambda x: -x, mean)), std=[1]),
    torchvision.transforms.Lambda(clip),
    torchvision.transforms.ToPILImage()
])

plt.figure()
plt.imshow(tf(img_pert.cpu()))
plt.show()