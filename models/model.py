from torchvision import transforms, models
from models.resnet import resnet34, resnet18
from models.simple_model import MNIST_Net, Cifar_Net


def get_model(model_name, device="cpu"):
    """
    loads the given model
    """

    if model_name == "Inceptionv3":
        net = models.inception_v3(pretrained=True)
    elif model_name == "Resnet34":
        net = resnet34(pretrained=True)
    elif model_name == "Resnet18":
        net = resnet18(pretrained=True)
    elif model_name == "VGG-19":
        net = models.vgg19(pretrained=True)
    elif model_name == "mnist_model":
        net = MNIST_Net()
        net.load_model()
    elif model_name == "cifar_model":
        net = Cifar_Net()
        net.load_model()
    else:
        raise ValueError

    net.to(device)
    net.eval()
    return net
