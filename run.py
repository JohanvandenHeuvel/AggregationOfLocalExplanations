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

import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


###########################
#  experiment conditions  #
###########################
params = {
    "model": "Resnet18_cifar10",
    "dataset": "cifar10",
    "batch_size": 40,
    "max_nr_batches": 50,  # -1 for no early stopping
    "attribution_methods": [
        "lime_1",
        "lime_2",
        "lime_3",
    ]
    + ["noise_uniform"] * 0,
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

attribution_params = {
    "lime_1": {"use_slic": True, "n_slic_segments": 10,},
    "lime_2": {"use_slic": True, "n_slic_segments": 100,},
    "lime_3": {"use_slic": True, "n_slic_segments": 1000,},
    "integrated_gradients": {"baseline": "black",},
    "deeplift": {},
    "gradientshap": {},
    "saliency": {},
    "occlusion": {},
    "smoothgrad": {},
    "guidedbackprop": {},
    "gray_image": {},
}

rbm_params = {
    "batch_size": 28,
    "learning_rate": 0.001,
    "n_iter": 100,
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
        dataset, batch_size=params["batch_size"], shuffle=True, num_workers=2
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
    # iter = dataloader.__iter__()
    # TODO: Remove later
    # for i in tqdm(range(len(dataloader))):
    #     (image_batch, label_batch) = next(iter)
    for i, (image_batch, label_batch) in tqdm(enumerate(dataloader)):

        # put data on gpu if possible
        image_batch = image_batch.to(device)
        label_batch = label_batch.to(device)

        # we use the predicted labels as targets
        label_batch = predict_label(model, image_batch).squeeze()

        # for what label the image should be explained for
        predicted_labels = predict_label(model, image_batch).squeeze()

        # only use images that the network predicts correctly
        mask = torch.eq(label_batch, predicted_labels)
        indices = torch.masked_select(torch.arange(0, len(mask)).to(device), mask)

        # generate explanations
        attributions = generate_attributions(
            image_batch[indices],
            label_batch[indices],
            model,
            params,
            attribution_params,
            device,
        )

        ###########################
        #  explanation processing #
        ###########################
        zero = torch.Tensor([0]).to(device)
        if params["attribution_processing"] == "filtering":
            # Set negative values to zero
            attributions = torch.max(attributions, zero)

            # Make sure we have values in range [0,1]
            attributions = normalize(params["normalization"], arr=attributions)

            ###########################
            #        ensembles        #
            ###########################

            ensemble_attributions = generate_ensembles(
                attributions, params["ensemble_methods"], rbm_params, device
            )

        elif params["attribution_processing"] == "splitting":
            # Split attributions in negative and positive parts
            pos_attr = torch.max(attributions, zero)
            neg_attr = torch.min(attributions, zero)

            # Make sure we have values in range [0,1]
            pos_norm, neg_norm = normalize(
                params["normalization"], arr=pos_attr, arr2=neg_attr
            )

            ###########################
            #        ensembles        #
            ###########################

            pos_ens = generate_ensembles(pos_norm, params["ensemble_methods"], device)
            neg_ens = generate_ensembles(neg_norm, params["ensemble_methods"], device)

            # Combine negative and positive attributions again
            ensemble_attributions = pos_ens - neg_ens

            # Finally also normalize individual attributions
            attributions = normalize(params["normalization"], arr=attributions)
        elif params['attribution_processing'] == 'none':
            # Make sure we have values in range [0,1]
            attributions = normalize(params["normalization"], arr=attributions)

            ###########################
            #        ensembles        #
            ###########################

            ensemble_attributions = generate_ensembles(
                attributions, params["ensemble_methods"], rbm_params, device
            )
        else:
            raise ValueError

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

        # # TODO: Uncommented, maybe put into a separate function
        # ###########################
        # #      plot examples      #
        # ###########################
        # for idx in range(len(indices)):
        #     # idx = 0  # first image of the batch
        #     original_img = (
        #         torch.mean(image_batch[indices], dim=1)[idx].cpu().detach().numpy()
        #     )
        #     images = [original_img]
        #
        #     # TODO don't plot noise attributions
        #     # one image for every attribution method
        #     for j in range(len(attributions)):
        #         attribution_img = attributions[j][idx].cpu().detach().numpy()
        #         images.append(attribution_img)
        #
        #     # one image for every ensemble method
        #     for j in range(len(params["ensemble_methods"])):
        #         ensemble_img = ensemble_attributions[j][idx].cpu().detach().numpy()
        #         images.append(ensemble_img)
        #
        #     my_plot(
        #         images,
        #         ["original"]
        #         + params["attribution_methods"]
        #         + params["ensemble_methods"]
        #         + ["flipped_rbm"],
        #         save=False,
        #     )

        print(i, params['max_nr_batches'])

        if i+1 >= params["max_nr_batches"] > 0:
            print("stopped early")
            break

    write_scores_to_file(scores)
    score_table = create_score_table(scores)
    pd.options.display.width = 0
    print(score_table)
    score_table_path = os.path.join(folder_path, "score_table.csv")
    score_table.to_csv(score_table_path)


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


def write_params_to_disk(params, name):
    file_name = "{}.json".format(name)
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

    ###########################
    #  for writing to disk    #
    ###########################
    results_dir = "results"

    now = datetime.datetime.now()
    folder_name = now.strftime("%m-%d_@%H-%M-%S")
    folder_path = os.path.join(results_dir, folder_name)
    os.makedirs(folder_path)

    write_params_to_disk(params, "params")
    write_params_to_disk(attribution_params, "attribution_params")
    write_params_to_disk(rbm_params, "rbm_params")

    main()
