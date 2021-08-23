import numpy as np

from normalize import normalize_sum1, normalize_abs, duplex_normalize_abs


def mean_ens(attributions):
    return np.mean(attributions, axis=1)


def variance_ens(attributions):
    epsilon = np.mean(attributions) * 10
    return np.mean(attributions, axis=1) / (np.std(attributions, axis=1) + epsilon)


def rbm_ens(rbms, attribution):
    for i, r in enumerate(rbms):
        r.fit(attribution)
        attribution = r.transform(attribution)
    return attribution


def compute_ensembles(attributions, noise_attributions, tasks, positive_filter):
    width, height = attributions.shape[1:]
    attributions = attributions.reshape(len(attributions), -1).T

    if noise_attributions is not None:
        noise_attributions = np.asarray(noise_attributions)
        noise_attributions = noise_attributions.reshape(len(noise_attributions), -1)

    ensemble_attributions = []
    for task in tasks:
        attr = attributions.copy()
        if task["nr_noise"] > 0:
            attr = np.concatenate((attr, noise_attributions[0:task["nr_noise"]]))

        if task["technique"] == "rbm" and not positive_filter:
            pos_attr, neg_attr = attr.copy(), attr.copy()
            pos_attr[pos_attr < 0] = 0
            neg_attr[neg_attr > 0] = 0

            attributions = np.array([duplex_normalize_abs(pos_attr[i], neg_attr[i]) for i in range(len(pos_attr))])
            pos_attr, neg_attr = (attributions[:, 0], attributions[:, 1])

            pos_ensemble = rbm_ens(task["rbm"], pos_attr)
            neg_ensemble = rbm_ens(task["rbm"], neg_attr)

            ensemble = pos_ensemble - neg_ensemble
        else:
            if positive_filter:
                attr[attr < 0] = 0
            attr = np.array([normalize_abs(a) for a in attr])

        if task["technique"] == "mean":
            ensemble = mean_ens(attr)
        elif task["technique"] == "var":
            ensemble = variance_ens(attr)
        elif task["technique"] == "rbm" and positive_filter:
            ensemble = rbm_ens(task["rbm"], attr)
        elif task["technique"] == "rbm" and not positive_filter:
            pass  # Case captured above
        else:
            raise ValueError("Ensemble technique not found")

        ensemble = ensemble.reshape(1, width, height)
        ensemble_attributions.append(ensemble)

    return np.stack(ensemble_attributions)




