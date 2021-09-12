import torchvision
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from abc import abstractmethod, ABC
import shap
import glob, os
from PIL import Image

# from torch_geometric.transforms import ToSLIC


def get_dataset(name, normalized=True, SLIC=False):

    if name == "mnist":
        return MNIST(normalized).dataset

    if name == "cifar10":
        return Cifar10(normalized, SLIC).dataset

    if name == "small_imagenet":
        return SmallImagenet().dataset

    if name == "imagenet":
        return ImageNet().dataset

    raise ValueError


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
            transform_list.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            )
        if SLIC:
            transform_list.append(ToSLIC(n_segments=75))

        self.transform = transforms.Compose(transform_list)


    @property
    def dataset(self):

        dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=self.transform
        )

        return dataset


def preprocess_image_tensor_imagenet(image):
    preprocess = transforms.Compose([
    ])


class ImageNet:
    def __init__(self, normalized=True):

        transform_list = [
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor()
        ]

        if normalized:
            transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.transform = torchvision.transforms.Compose(transform_list)

    @property
    def dataset(self):
        dataset = ImageNetDataset(self.transform)
        return dataset


class ImageNetDataset(Dataset):
    def __init__(self, transform):
        self.directory_name = os.path.abspath(__file__ + "/../../imagenet")

        os.chdir(f"{self.directory_name}")
        self.files = glob.glob("*.jpg")

        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        raw_image = Image.open(f"{self.directory_name}/{self.files[idx]}")
        image_tensor = self.transform(raw_image)
        return image_tensor, -1


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


