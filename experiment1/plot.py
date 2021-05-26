import matplotlib.pyplot as plt
import numpy as np
from data import get_data
import torchvision


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def plot_batch(images, labels, classes):
    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    batch_size = len(images)
    print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))


def plot_training_batch(data_set):
    batch_size = 4
    trainloader, _, classes = get_data(data_set, batch_size)
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    plot_batch(images, labels, classes)


if __name__=='__main__':
    plot_training_batch('MNIST')