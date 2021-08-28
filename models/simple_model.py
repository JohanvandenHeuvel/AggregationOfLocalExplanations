import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, models
from torchvision.models import resnet18
from models.train import train
from scripts.load_data import get_data
from models.resnet import resnet34, resnet18


class MNIST_Net(nn.Module):
    """
    Simple CNN models for MNIST dataset
    """

    def __init__(self, model_name="mnist_net", device="cpu"):
        super(MNIST_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

        self.model_name = model_name
        self.device = device

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def save_model(self, path=None):
        if path is None:
            path = self.model_name
        path = path + ".pt"
        torch.save(self.state_dict(), path)
        print("models saved")

    def load_model(self, path=None):
        try:
            if path is None:
                path = self.model_name
            self.load_state_dict(torch.load(path))
            print("models loaded")
        except:
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081))]
            )
            trainset = torchvision.datasets.MNIST(
                root="./data", train=True, download=True, transform=transform
            )
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=64, shuffle=True, num_workers=2
            )
            train(self, trainloader)


class Cifar_Net(nn.Module):
    """
    Simple CNN models for CIFAR10
    """

    def __init__(self, model_name="cifar_net"):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.model_name = model_name

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def save_model(self, path=None):
        if path is None:
            path = self.model_name
        path = path + ".pt"
        torch.save(self.state_dict(), path)
        print("models saved")

    def load_model(self, path=None):
        try:
            if path is None:
                path = self.model_name
            self.load_state_dict(torch.load(path))
            print("models loaded")
        except:
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081))]
            )

            trainset = torchvision.datasets.CIFAR10(
                root="./data", train=True, download=True, transform=transform
            )
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=64, shuffle=True, num_workers=2
            )
            train(self, trainloader)




if __name__ == "__main__":
    net = Cifar_Net()
    trainloader, testloader, classes = get_data()
    train(net, trainloader)
