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

    # if name == "small_imagenet":
    #     return SmallImagenet()


def image_3D_to_4D(image):
    return image.reshape(-1, image.size()[0], image.size()[1], image.size()[2])


def image_4D_to_3D(image):
    return image.reshape(-1, image.size()[2], image.size()[3])


# class Dataset:
#     def __init__(self):
#         self.dataset = None
#
#     @abstractmethod
#     def preprocess_pil(self, image):
#         pass
#
#     @abstractmethod
#     def preprocess_tensor(self, image):
#         pass
#
#     # def get_raw_image(self, idx):
#     #     img, label = self.dataset[idx]
#     #     return img, label
#
#     # def get_image(self, idx):
#     #     raw_image, _ = self.dataset[idx]
#     #     return self.preprocess_tensor(raw_image).to(self.device)


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


# class SmallImagenet(Dataset, ABC):
#     def __init__(self):
#         super().__init__()
#
#         # TODO get in same form as cifar10
#
#         self.mean = [0.485, 0.456, 0.406]
#         self.std = [0.229, 0.224, 0.225]
#
#         self.dataset, _ = shap.datasets.imagenet50()
#         self.dataset /= 255
#
#         # self.transform = transforms.Compose(
#         #     [
#         #         transforms.ToTensor(),
#         #         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#         #     ]
#         # )
#
#     def normalize(self, image):
#         if image.max() > 1:
#             image /= 255
#         image = (image - self.mean) / self.std
#         # in addition, roll the axis so that they suit pytorch
#         return torch.tensor(image.swapaxes(-1, 1).swapaxes(2, 3)).float()
#
#     def get_image(self, idx):
#         X = self.dataset[idx]
#         return self.normalize(X).to(self.device)
