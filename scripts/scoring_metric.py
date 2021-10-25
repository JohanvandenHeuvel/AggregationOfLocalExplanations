import numpy as np
import torch


def compute_batch_score(scores,
                        measure,
                        images,
                        labels,
                        attributions,
                        ensemble_attributions,
                        params):
    attributions = torch.cat((attributions, ensemble_attributions))
    attributions = attributions.permute((1, 0, 2, 3))

    attribution_titles = np.array(list(next(iter(scores.values())).keys()))

    # Filter out noise
    indices = np.where(((attribution_titles != "noise_normal") &
                        (attribution_titles != "noise_uniform")))[0]
    attributions = attributions[:, indices]
    attribution_titles = attribution_titles[indices]

    IAUC = "IAUC" in params["scoring_methods"]
    DAUC = "DAUC" in params["scoring_methods"]
    IROF = "IROF" in params["scoring_methods"]

    measure_result = measure.compute_batch(images,
                                           attributions,
                                           labels,
                                           IAUC=IAUC,
                                           DAUC=DAUC,
                                           IROF=IROF)

    for scoring_method in measure_result.keys():
        for i, attr_title in enumerate(attribution_titles):
            new_scores = measure_result[scoring_method][0][:, i]
            new_scores = list(new_scores.detach().cpu().numpy())
            scores[scoring_method][attr_title] += new_scores
