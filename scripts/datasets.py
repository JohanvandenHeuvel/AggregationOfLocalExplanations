import torchvision
import torch
from torchvision import transforms
from abc import abstractmethod, ABC
import shap


def get_dataset(name):

    if name == "mnist":
        return MNIST()

    if name == "cifar10":
        return Cifar10()

    if name == "small_imagenet":
        return SmallImagenet()

    raise ValueError


def image_3D_to_4D(image):
    return image.reshape(-1, image.size()[0], image.size()[1], image.size()[2])


def image_4D_to_3D(image):
    return image.reshape(-1, image.size()[2], image.size()[3])



def MNIST():

    transform = torchvision.transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    return dataset


def Cifar10():

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
    )

    dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    return dataset


def SmallImagenet():
    # https://shap.readthedocs.io/en/latest/example_notebooks/image_examples/image_classification/Explain%20an%20Intermediate%20Layer%20of%20VGG16%20on%20ImageNet%20%28PyTorch%29.html
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def normalize(image):
        if image.max() > 1:
            image /= 255
        image = (image - mean) / std
        # in addition, roll the axis so that they suit pytorch
        return torch.tensor(image.swapaxes(-1, 1).swapaxes(2, 3)).float()

    X, y = shap.datasets.imagenet50()
    X = normalize(X)
    y = torch.Tensor(y).unsqueeze(1) # labels are actually not usable

    dataset = torch.utils.data.TensorDataset(X, y)

    return dataset


