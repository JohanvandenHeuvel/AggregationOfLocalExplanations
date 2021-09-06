import torchvision
import torch
from torchvision import transforms
from abc import abstractmethod, ABC
import shap

from torch_geometric.transforms import ToSLIC


def get_dataset(name, normalized=True, SLIC=False):

    if name == "mnist":
        return MNIST(normalized).dataset

    if name == "cifar10":
        return Cifar10(normalized, SLIC).dataset

    if name == "small_imagenet":
        return SmallImagenet().dataset

    raise ValueError


def get_tensor_transform(name):

    if name == "mnist":
        return MNIST().transform

    if name == "cifar10":
        return Cifar10().transform

    raise ValueError


def get_pil_transform(name):

    if name == "cifar10":

        def f(image):
            return image

        return f


def image_3D_to_4D(image):
    return image.reshape(-1, image.size()[0], image.size()[1], image.size()[2])


def image_4D_to_3D(image):
    return image.reshape(-1, image.size()[2], image.size()[3])


class MNIST:
    def __init__(self, normalized=True):

        if normalized:
            self.transform = torchvision.transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )
        else:
            self.transform = torchvision.transforms.Compose([transforms.ToTensor()])

    @property
    def dataset(self):

        dataset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=self.transform
        )

        return dataset


class Cifar10:
    def __init__(self, normalized=True, SLIC=False):

        transform_list = [transforms.ToTensor()]
        if normalized:
            transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        if SLIC:
            transform_list.append(ToSLIC(n_segments=75))

        self.transform = transforms.Compose(transform_list)

        # if normalized:
        #     self.transform = transforms.Compose(
        #         [
        #             transforms.ToTensor(),
        #         ]
        #     )
        # else:
        #     self.transform = transforms.Compose([transforms.ToTensor()])

    @property
    def dataset(self):

        dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=self.transform
        )

        return dataset




class SmallImagenet:
    def __init__(self):
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
        y = torch.Tensor(y).unsqueeze(1)  # labels are actually not usable

        self.dataset = torch.utils.data.TensorDataset(X, y)
