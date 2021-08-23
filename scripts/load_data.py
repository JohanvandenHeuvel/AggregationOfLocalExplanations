import numpy as np
from numpy import genfromtxt
import torch
from torchvision import transforms
from PIL import Image
import torchvision

DOWNLOAD_DATASET = False

# TODO: Remove
def get_data(data_set="CIFAR10", batch_size=4):

    if data_set == "CIFAR10":
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



def cifar10_preprocess_tensor(image):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transform(image).double()


def mnist_preprocess_pil(image):
    return image


def mnist_preprocess_tensor(image):
    if not isinstance(image, torch.Tensor):
        transform = transforms.ToTensor()
        image = transform(image)
    imageT = image.float()
    imageT = ((imageT / 255) - 0.1307) / 0.3081
    imageT = imageT.reshape(1, 28, 28).double()
    return imageT


def get_image_preprocessors(dataset):
    if dataset == "mnist":
        return None, mnist_preprocess_tensor
    elif dataset == "cifar10":
        return None, cifar10_preprocess_tensor


def load_image(dataset, image_id, device):
    if dataset == "mnist":
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=DOWNLOAD_DATASET)

        raw_image = testset.data[image_id].reshape(1, 28, 28)
        image_tensor = mnist_preprocess_tensor(raw_image)
        label = testset.targets[image_id].item()
    elif dataset == "cifar10":
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=DOWNLOAD_DATASET)

        raw_image = testset.data[image_id]
        image_tensor = cifar10_preprocess_tensor(raw_image)
        label = testset.targets[image_id]
    else:
        raise ValueError

    image_tensor.to(device)
    return image_tensor, raw_image, label


def image_3D_to_4D(image):
    return image.reshape(-1, image.size()[0], image.size()[1], image.size()[2])


def image_4D_to_3D(image):
    return image.reshape(-1, image.size()[2], image.size()[3])
