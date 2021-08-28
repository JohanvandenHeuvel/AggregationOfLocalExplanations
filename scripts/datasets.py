import torchvision
import torch
from torchvision import transforms
from abc import abstractmethod, ABC
import shap


def get_dataset(name, device):

    if name == "mnist":
        return MNIST(device)

    if name == "cifar10":
        return Cifar10(device)

    if name == "small_imagenet":
        return SmallImagenet(device)


def image_3D_to_4D(image):
    return image.reshape(-1, image.size()[0], image.size()[1], image.size()[2])


def image_4D_to_3D(image):
    return image.reshape(-1, image.size()[2], image.size()[3])


class Dataset:
    def __init__(self, device):
        self.device = device
        self.dataset = None

    @abstractmethod
    def preprocess_pil(self, image):
        pass

    @abstractmethod
    def preprocess_tensor(self, image):
        pass

    def get_raw_image(self, idx):
        return self.dataset.data[idx].to(self.device), self.dataset.targets[idx]

    def get_image(self, idx):
        raw_image, _ = self.get_raw_image(idx)
        return self.preprocess_tensor(raw_image).to(self.device)


class MNIST(Dataset, ABC):
    def __init__(self, device):
        super().__init__(device)

        self.dataset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True
        )

    def preprocess_tensor(self, image):
        if not isinstance(image, torch.Tensor):
            transform = transforms.ToTensor()
            image = transform(image)
        imageT = image.float()
        imageT = ((imageT / 255) - 0.1307) / 0.3081
        imageT = imageT.reshape(1, 28, 28).double()
        return imageT

    def preprocess_pil(self, image):
        return image


class Cifar10(Dataset, ABC):
    def __init__(self, device):
        super().__init__(device)

        self.dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True
        )

    def preprocess_tensor(self, image):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        return transform(image).double()


class SmallImagenet(Dataset, ABC):
    def __init__(self, device):
        super().__init__(device)

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.dataset, _ = shap.datasets.imagenet50()
        self.dataset /= 255

    def normalize(self, image):
        if image.max() > 1:
            image /= 255
        image = (image - self.mean) / self.std
        # in addition, roll the axis so that they suit pytorch
        return torch.tensor(image.swapaxes(-1, 1).swapaxes(2, 3)).float()

    def get_image(self, idx):
        X = self.dataset[idx]
        return self.normalize(X).to(self.device)
