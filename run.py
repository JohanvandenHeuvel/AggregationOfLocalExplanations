import torch

from models.model import get_model
import json
from tqdm import tqdm
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scripts.run_experiment import *
from scripts.attribution_methods import attribution_method, generate_attributions
from scripts.normalize import normalize
import scripts.datasets as datasets
from scripts.ensemble import generate_ensembles
from scripts.scoring_measures import calc_scores

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

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
    "model": "mnist_model",
    "dataset": "mnist",
    "batch_size": 10,
    "attribution_methods": ["deeplift", "saliency"]
    + ["noise_uniform"] * 0,
    "ensemble_methods": ["mean", "variance", "rbm"],
    "attribution_processing": "filtering",
    "normalization": "min_max",
    "scores": ["insert", "delete", "irof"],  # TODO: New params have been added
    "scores_btach_size": 80,
    "package_size": 2,
    "irof_segments": 60,
    "irof_sigma": 4
}


###########################
#    running the code     #
###########################
def main():
    # classification model
    model = get_model(params["model"], device=device)

    # # methods for explaining
    # attribution_methods = [
    #     attribution_method(method, model) for method in params["attribution_methods"]
    # ]

    # dataset and which images to explain the classification for
    dataset = datasets.get_dataset(params["dataset"])
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=params["batch_size"], shuffle=False, num_workers=2
    )


    scores = dict([(score, dict([(m, []) for m in params["attribution_methods"]])) for score in params["scores"]])

    # TODO: I needed to change this for loop, beacuse PyCharm made a lot of problems with it. We can add it later again.
    # for i, (image_batch, label_batch) in tqdm(enumerate(dataloader)):
    for i in tqdm(range(dataloader.__len__())):
        image_batch, label_batch = next(iter(dataloader))

        # put data on gpu if possible
        image_batch = image_batch.to(device)
        label_batch = label_batch.to(device)

        # TODO remove (small imagenet does not have it's own labels)
        if params["dataset"] == "small_imagenet":
            label_batch = predict_label(model, image_batch).squeeze()

        # for what label the image should be explained for
        predicted_labels = predict_label(model, image_batch).squeeze()

        # only use images that the network predicts correctly
        mask = torch.eq(label_batch, predicted_labels)
        indices = torch.masked_select(torch.arange(0, len(mask)).to(device), mask)

        # generate explanations
        attributions = generate_attributions(
            image_batch[indices], label_batch[indices], model, params["attribution_methods"], device,
        )

        # TODO: Integrate it nicely, e.g. attributions & ensembles need to be handed over
        calc_scores(model, image_batch, label_batch, attributions, params["attribution_methods"], scores,
                    params["scores_btach_size"], params["package_size"], params["irof_segments"], params["irof_sigma"],
                    device)
        if i == 10:  # TODO: Integrate
            create_statistics_table(scores)
            return


    for i in range(10):

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
        for idx in range(8):
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


def create_statistics_table(scores):
    # For each method add mean and std column
    columns = [[method + " mean", method + " std"] for method in scores.keys()]
    columns = sum(columns, [])
    columns = ["method"] + columns

    data = []
    for method in scores[list(scores.keys())[0]].keys():
        data.append([method])

    for statistic in scores.keys():
        for j, method in enumerate(scores[statistic]):
            data[j].append(np.mean(scores[statistic][method]))
            data[j].append(np.std(scores[statistic][method]))

    df = pd.DataFrame(data, columns=columns)
    return df


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
