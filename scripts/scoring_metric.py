import numpy as np
import abc
import torch
from sklearn.metrics import auc

from models.predict import calculate_probs
from scripts.irof import IrofDataset
from scripts.pixel_relevancy import PixelRelevancyDataset
from scripts.causual_metric import CausalMetric


class ScoringMetric:
    def __init__(self, model, scores, params):
        self._model = model
        self._params = params
        self._scores = scores

    def compute_batch_score(self, image_batch, label_batch, attributions, ensembles):
        if len(self._scores.keys()) == 0:
            return

        attr_titles = list(next(iter(self._scores.values())).keys())
        all_attributions = torch.cat((attributions, ensembles), 0)

        for i, (image, label) in enumerate(zip(image_batch, label_batch)):
            for scoring_method in self._params["scoring_methods"]:
                for attr, title in zip(all_attributions[:, i], attr_titles):
                    if title == "noise_normal" or title == "noise_uniform":
                        continue

                    scoring_dataset = self._make_scoring_dataset(
                        scoring_method, image, attr
                    )
                    score, _ = self._calc_score(scoring_method, scoring_dataset, label)
                    self._scores[scoring_method][title].append(score)

    def _make_scoring_dataset(self, scoring_method, image, attr):
        device = image.device

        if scoring_method == "insert":
            dataset = PixelRelevancyDataset(
                image, attr, True, self._params["scores_batch_size"],
                self._params["package_size"], device
            )
        elif scoring_method == "delete":
            dataset = PixelRelevancyDataset(
                image, attr, False, self._params["scores_batch_size"],
                self._params["package_size"], device
            )
        elif scoring_method == "irof":
            dataset = IrofDataset(
                image, attr, self._params["scores_batch_size"],
                self._params["irof_segments"], self._params["irof_sigma"],
                device
            )
        else:
            raise ValueError

        return dataset

    def _calc_score(self, scoring_method, scoring_dataset, label):

        probs = []
        for j, img_batch in enumerate(scoring_dataset):
            probs += [calculate_probs(self._model, img_batch)[:, label]]

        probs = torch.cat(probs)
        rel_probs = probs[:-1] / probs[-1]

        x = np.arange(0, len(rel_probs))
        y = rel_probs.detach().cpu().numpy()

        if scoring_method == "irof":
            y = 1-y

        score = auc(x, y) / len(rel_probs)

        return score, y
