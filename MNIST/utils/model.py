import torch
import torch.nn as nn

class CNN_0(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout2d(0.2)
        )
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25088, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.dense(x)
        return x

class LeNet(nn.Module):
    def __init__(self, classes=10) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.full = nn.Sequential(
            nn.Flatten(),
            nn.Linear(400, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.full(x)
        return x

if __name__ == '__main__':
    net = LeNet()
    inputs = torch.ones((1, 1, 28, 28))
    print(inputs.shape)
    outputs = net(inputs)
    print(outputs.shape)