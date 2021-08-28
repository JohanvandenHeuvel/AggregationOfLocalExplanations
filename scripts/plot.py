import numpy as np
import matplotlib.pyplot as plt
import torchvision

# from scripts.load_data import get_data


# # functions to show an image
# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
#
# def plot_batch(images, labels, classes):
#     # show images
#     imshow(torchvision.utils.make_grid(images))
#     # print labels
#     batch_size = len(images)
#     print(" ".join("%5s" % classes[labels[j]] for j in range(batch_size)))
#
#
# def plot_training_batch(data_set):
#     batch_size = 4
#     trainloader, _, classes = get_data(data_set, batch_size)
#     # get some random training images
#     dataiter = iter(trainloader)
#     images, labels = dataiter.next()
#
#     plot_batch(images, labels, classes)


def plot_explanations(
    explanations, titles, figsize,
):
    plt.figure(figsize=figsize)
    for i in range(len(titles)):
        plt.subplot(1, len(titles), i + 1)
        plt.imshow(explanations[i], cmap="OrRd")
        plt.title(titles[i])
        plt.colorbar()
    plt.show()


def prepare_colorchannel(image):
    if len(image.shape) == 2:
        return image
    if image.shape[0] == 1:
        return image.reshape(image.shape[1], image.shape[2])
    else:
        return image.transpose(1, 2, 0)


def plot_difference(ens1, ens2):
    diff = ens1 - ens2
    diff_pos = diff.copy()
    diff_neg = diff.copy()
    diff_pos[diff_pos < 0] = 0
    diff_neg[diff_neg > 0] = 0

    plot_diff = [diff, diff_pos, -diff_neg]
    titles = ["Difference", "Positive Difference", "Negative Difference"]

    plt.figure(figsize=(25, 5))
    plot_explanations(plot_diff, titles, figsize=(10, 2.2))
    # for i in range(len(titles)):
    #     plt.subplot(1, len(titles), i+1)
    #     plt.imshow(plot_diff[i])
    #     plt.title(titles[i])
    #     plt.colorbar()
    # plt.show()


def prepare_image_for_showing(image):
    diff = np.max(image) - np.min(image)
    img = (image - np.min(image)) * 255 / diff
    img = prepare_colorchannel(img)
    return (img).astype(np.uint8)


def plot_orig_images(image, annotation=None):
    plt.figure(figsize=(5, 2.2))

    plt.subplot(1, 2, 1)
    plt.imshow(prepare_image_for_showing(image.cpu().detach().numpy()), cmap="gray")
    plt.title("Original image")

    if annotation is not None:
        an_normalized = annotation - np.min(annotation)
        plt.subplot(1, 2, 2)
        plt.imshow(prepare_image_for_showing(annotation))
        plt.title("Annotation")

    plt.show()


def plot_rbm_weight_matrix(weight_matrix, methods):
    nr_noise_explanations = len(weight_matrix[0]) - len(methods)
    rbm_weights = np.asarray(weight_matrix)
    colors = ["b", "g", "r", "c", "y", "k"]

    x = list(np.arange(0, len(rbm_weights[0])))
    y = rbm_weights.mean(axis=0)
    std = rbm_weights.std(axis=0)

    print("RBM weights mean: ", y)
    print("RBM weights std: ", std)

    for i in range(len(methods)):
        plt.plot(x[i], y[i], marker="o", color=colors[i])
        plt.errorbar(x[i], y[i], std[i], fmt="", label=methods[i], color=colors[i])
    if len(methods) < len(x):
        plt.plot(x[len(methods) :], y[len(methods) :], marker="o", color=colors[-1])
        plt.errorbar(
            x[len(methods) :],
            y[len(methods) :],
            std[len(methods) :],
            fmt="",
            label="Random",
            color=colors[-1],
        )
    plt.legend()
    plt.title("RBM weights")
    plt.xticks(x, "")

    plt.show()


# if __name__ == "__main__":
#     plot_training_batch("MNIST")
