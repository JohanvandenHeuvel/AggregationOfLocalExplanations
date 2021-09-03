import torch
import json
from tqdm import tqdm
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from models.model import get_model
from scripts.attribution_methods import attribution_method, generate_attributions
from captum.attr import visualization as viz
from scripts.normalize import normalize
import scripts.datasets as datasets
from scripts.ensemble import generate_ensembles
from models.predict import predict_label
from scripts.scoring_metric import ScoringMetric

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
    "model": "Resnet18_cifar10",
    # "model": "mnist_model",
    "dataset": "cifar10",
    "batch_size": 10,
    "max_nr_batches": 1,
    "attribution_methods": [
        "gradientshap",
        "deeplift",
        "lime",
        "saliency",
        "occlusion",
        "smoothgrad",
        "guidedbackprop",
        "gray_image",
    ]
    + ["noise_uniform"] * 0,
    # "attribution_methods": ["lime"] + ["noise_uniform"] * 0,
    "ensemble_methods": [
        "mean",
        "variance",
        "rbm",
        "flipped_rbm",
        "rbm_flip_detection",
    ],
    "attribution_processing": "filtering",
    "normalization": "min_max",
    "scoring_methods": ["insert", "delete", "irof"],
    "scores_batch_size": 100,
    "package_size": 1,
    "irof_segments": 60,
    "irof_sigma": 4,
}


###########################
#    running the code     #
###########################
def main():
    # classification model
    model = get_model(params["model"], device=device)

    # dataset and which images to explain the classification for
    dataset = datasets.get_dataset(params["dataset"])
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=params["batch_size"], shuffle=False, num_workers=2
    )

    # Preparation for score computation later
    attr_titles = params["attribution_methods"] + params["ensemble_methods"]
    scores = dict(
        [
            (score, dict([(m, []) for m in attr_titles]))
            for score in params["scoring_methods"]
        ]
    )
    metric = ScoringMetric(model, scores, params)

    # TODO: Remove later
    # for i in tqdm(range(len(dataloader))):
    #     (image_batch, label_batch) = next(dataloader.__iter__())
    for i, (image_batch, label_batch) in tqdm(enumerate(dataloader)):

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
            image_batch[indices], label_batch[indices], model, params, device,
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

        # make sure it sums to 1
        ensemble_attributions = normalize(
            params["normalization"], arr=ensemble_attributions
        )

        ###########################
        #       statistics        #
        ###########################

        metric.compute_batch_score(
            image_batch[indices],
            label_batch[indices],
            attributions,
            ensemble_attributions,
        )

        # TODO: Uncommented, maybe put into a separate function
        ###########################
        #      plot examples      #
        ###########################
        for idx in range(len(indices)):
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
                + params["ensemble_methods"]
                + ["flipped_rbm"],
                save=False,
            )

        if i + 1 >= params["max_nr_batches"]:
            write_scores_to_file(scores)
            score_table = create_score_table(scores)
            pd.options.display.width = 0
            print(score_table)
            return


def create_score_table(scores):
    # For each method add mean and std column
    columns = [[method + " mean", method + " std"] for method in scores.keys()]
    columns = sum(columns, [])
    columns = ["method"] + columns

    # Create a row for each attribution method
    data = []
    method_titles = list(scores[list(scores.keys())[0]].keys()) + ["rbm_ideal"]
    data = [[method] for method in method_titles]

    # Calculate for each combination of attribution and method
    # the mean and variance score
    for statistic in scores.keys():
        for j, method in enumerate(scores[statistic]):
            data[j].append(np.mean(scores[statistic][method]))
            data[j].append(np.std(scores[statistic][method]))

        # Compute rbm_ideal
        # For every score pair of rbm and rbm_flipped, pick the better one.
        function = np.min if statistic == "delete" else np.max
        rbms = np.asarray([scores[statistic]["rbm"], scores[statistic]["flipped_rbm"]])
        best_rbm_scores = function(rbms, axis=0)
        data[-1].append(np.mean(best_rbm_scores))
        data[-1].append(np.std(best_rbm_scores))

    df = pd.DataFrame(data, columns=columns)
    return df


def my_plot(images, titles, save=False):
    # make a square
    x = int(np.ceil(np.sqrt(len(images))))
    fig, axs = plt.subplots(x, x, figsize=(10, 10))

    # plot the images
    for i, ax in enumerate(axs.flatten()):
        if i < len(images):
            viz.visualize_image_attr(
                images[i][..., np.newaxis],
                # attr_map.permute(1, 2, 0).numpy(),  # adjust shape to height, width, channels
                method="heat_map",
                sign="all",
                show_colorbar=True,
                title=titles[i],
                plt_fig_axis=(fig, ax),
                use_pyplot=False,
            )
        else:
            ax.set_visible(False)

    plt.tight_layout()

    if save:
        file_name = now.strftime("%m-%d_@%H-%M-%S.png")
        file_path = os.path.join(folder_path, file_name)
        fig.savefig(file_path, dpi=fig.dpi)
    else:
        plt.show()


def show_attr(attr_map, title, plot_loc):
    viz.visualize_image_attr(
        attr_map[..., np.newaxis],
        # attr_map.permute(1, 2, 0).numpy(),  # adjust shape to height, width, channels
        method="heat_map",
        sign="all",
        show_colorbar=True,
        title=title,
        plt_fig_axis=plot_loc,
    )


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


def write_scores_to_file(scores):
    file_name = "scores.npy"
    file_path = os.path.join(folder_path, file_name)
    np.save(file_path, scores)


if __name__ == "__main__":
    write_params_to_disk()
    main()
