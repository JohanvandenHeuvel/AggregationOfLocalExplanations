import torch
import torchvision
import torchvision.transforms as transforms


def get_data(data_set="CIFAR10", batch_size=4):

    if data_set=="CIFAR10":
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    elif data_set == "MNIST":
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081))])
        trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                                download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False,
                                               download=True, transform=transform)
        classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    else:
        raise ValueError('data set not recognized')

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)


    return trainloader, testloader, classes


