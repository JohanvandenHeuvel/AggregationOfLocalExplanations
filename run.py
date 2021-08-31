import torch

from scripts.run_experiment import *
from scripts.attribution_methods import attribution_method, generate_attributions
from scripts.normalize import normalize
import scripts.datasets as datasets
from scripts.ensemble import generate_ensembles
from models.model import get_model
import json
from tqdm import tqdm

import os
import datetime

import numpy as np

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###########################
#  for writing to disk    #
###########################
results_dir = "results"
now = datetime.datetime.now()
folder_name = now.strftime("%m-%d_@%H-%M-%S")
folder_path = os.path.join(results_dir, folder_name)
os.makedirs(folder_path)

###########################
#  experiment conditions  #
###########################
params = {
    "model": "Resnet18_cifar10",
    "dataset": "cifar10",
    "batch_size": 1,
    "attribution_methods": ["deeplift", "smoothgrad", "saliency"]
    + ["noise_uniform"] * 0,
    "ensemble_methods": ["mean", "variance", "rbm"],
    "attribution_processing": "filtering",
    "normalization": "min_max",
}


###########################
#    running the code     #
###########################
def main():
    # classification model
    model = get_model(params["model"], device=device)

    # methods for explaining
    attribution_methods = [
        attribution_method(method, model) for method in params["attribution_methods"]
    ]

    # dataset and which images to explain the classification for
    dataset = datasets.get_dataset(params["dataset"])
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=params["batch_size"], shuffle=False, num_workers=2
    )

    for i, (image_batch, label_batch) in tqdm(enumerate(dataloader)):

        # put data on gpu if possible
        image_batch = image_batch.to(device)
        label_batch = label_batch.to(device)

        # for what label the image should be explained for
        predicted_labels = predict_label(model, image_batch)
        predicted_labels = predicted_labels.squeeze()

        # only use images that the network predicts correctly
        mask = torch.eq(label_batch, predicted_labels)
        indices = torch.masked_select(torch.arange(0, len(mask)).to(device), mask)

        # generate explanations
        attributions = generate_attributions(
            image_batch[indices], label_batch[indices], attribution_methods, device,
        )

        ###########################
        #  explanation processing #
        ###########################
        # remove negative values in some way
        if (
            params["attribution_processing"] == "filtering"
        ):  # set negative values to zero
            attributions = torch.max(attributions, torch.Tensor([0]).to(device))
        if (
            params["attribution_processing"] == "splitting"
        ):  # split attributions in negative and positive parts
            # TODO
            pass

        # make sure it sums to 1
        attributions = normalize(params["normalization"], arr=attributions)

        ###########################
        #        ensembles        #
        ###########################

        ensemble_attributions = generate_ensembles(
            attributions, params["ensemble_methods"], device
        )

        ###########################
        #      plot examples      #
        ###########################
        for idx in range(1):
            # idx = 0  # first image of the batch
            original_img = (
                torch.mean(image_batch[indices], dim=1)[idx].cpu().detach().numpy()
            )
            images = [original_img]

            # TODO don't plot noise attributions
            # one image for every attribution method
            for j in range(len(params["attribution_methods"])):
                attribution_img = attributions[j][idx].cpu().detach().numpy()
                images.append(attribution_img)

            # one image for every ensemble method
            for j in range(len(params["ensemble_methods"])):
                ensemble_img = ensemble_attributions[j][idx].cpu().detach().numpy()
                images.append(ensemble_img)

            my_plot(
                images,
                ["original"]
                + params["attribution_methods"]
                + params["ensemble_methods"],
                save=False,
            )

        break


def my_plot(images, titles, save=False):
    # make a square
    x = int(np.ceil(np.sqrt(len(images))))
    fig, axs = plt.subplots(x, x, figsize=(10, 10))

    # plot the images
    for i, ax in enumerate(axs.flatten()):
        if i < len(images):
            im = ax.imshow(images[i])
            ax.set_title(titles[i])
            fig.colorbar(im, ax=ax)
        else:
            ax.set_visible(False)

    if save:
        file_name = now.strftime("%m-%d_@%H-%M-%S.png")
        file_path = os.path.join(folder_path, file_name)
        fig.savefig(file_path, dpi=fig.dpi)
    else:
        plt.show()


def write_params_to_disk():
    file_name = "params.json"
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, "w") as f:
        json.dump(params, f, indent=4)


def write_to_file(my_str):
    file_name = "test.txt"
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, "w") as f:
        f.write(my_str)


if __name__ == "__main__":
    write_params_to_disk()
    main()
