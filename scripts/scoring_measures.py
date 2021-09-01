import numpy as np
import abc
import torch
from sklearn.metrics import auc

from models.predict import calculate_probs
from scripts.irof import IrofDataset
from scripts.pixel_relevancy import PixelRelevancyDataset


def calc_scores(
    model, image_batch, label_batch, attributions, scores, params,
):
    for i, (image, label) in enumerate(zip(image_batch, label_batch)):
        for scoring_method in params["scoring_methods"]:
            for attr, title in zip(attributions[:, i], params["attribution_methods"]):
                scoring_dataset = make_scoring_dataset(
                    scoring_method, image, attr, params
                )
                score = calc_score(model, scoring_dataset, label,)
                scores[scoring_method][title].append(score)


# def calc_score(scoring_methods, model, image, label, attributions, params):
#     """
#     compute score for a single image
#     """


def make_scoring_dataset(scoring_method, image, attr, params):
    batch_size = params["scores_batch_size"]
    package_size = params["package_size"]
    irof_segments = params["irof_segments"]
    irof_sigma = params["irof_sigma"]

    device = image.device

    if scoring_method == "insert":
        dataset = PixelRelevancyDataset(
            image, attr, True, batch_size, package_size, device
        )
    elif scoring_method == "delete":
        dataset = PixelRelevancyDataset(
            image, attr, False, batch_size, package_size, device
        )
    elif scoring_method == "irof":
        dataset = IrofDataset(
            image, attr, batch_size, irof_segments, irof_sigma, device
        )
    else:
        raise ValueError

    return dataset


def calc_score(model, scoring_dataset, label):

    probs = []
    for img_batch in scoring_dataset:
        probs += [calculate_probs(model, img_batch)[:, label]]

    probs = torch.cat(probs)
    rel_probs = probs / probs[-1]

    x = np.arange(0, len(rel_probs))
    y = rel_probs.detach().cpu().numpy()
    score = auc(x, y) / len(rel_probs)

    return score
