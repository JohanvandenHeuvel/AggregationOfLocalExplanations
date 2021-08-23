import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neural_network import BernoulliRBM
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pandas as pd

from load_data import load_image, get_image_preprocessors
from scoring_measures import pixel_score, irof_score
from explanation import compute_explanations, compute_noise_attributions
from ensemble import compute_ensembles
from simple_model.model import load_model
from statistics_record import AttrbutionsRecord
from normalize import normalize_abs

PACKAGE_SIZE = 3  # TODO: Change to 1000
PIXEL_SCORE_PIXEL_GROUP_SIZE = 20
BATCH_SIZE = 20
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")

import warnings
warnings.filterwarnings('ignore')


def predict_label(net, image):
    net = net.double()
    output = net(torch.unsqueeze(image.double(), 0))
    output = F.softmax(output, dim=1)
    _, label = torch.topk(output, 1)
    return label.item()


def seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_filenames(title, experiment):
    directory = AttrbutionsRecord.get_directory(title, "individual_attribution")
    files = os.listdir(directory)
    return files


def print_image(image, text):
    print(text)
    if len(image.shape) == 2:
        plt.imshow(image)
    else:
        plt.imshow(image.transpose(1, 2, 0))
    plt.colorbar()
    plt.show()


def generate_explanations(title, dataset, methods, nr_images, model_name=None):
    model = load_model(dataset, model_name).to(device)
    pil_preprocessor, tensor_preprocessor = get_image_preprocessors(dataset)

    image_counter = 0
    image_id = 0
    ar = None
    pbar = tqdm(total=nr_images)
    while image_counter < nr_images:
        if ar is None:
            ar = AttrbutionsRecord(title, dataset, int(np.floor((image_counter+1) / PACKAGE_SIZE)), methods, model_name)
        seed(image_id)
        image, raw_image, true_label = load_image(dataset, image_id, device)
        image_id += 1

        if predict_label(model, image) != true_label:
            continue
        ar.record_image(image_id, true_label, image, raw_image)

        attributions = compute_explanations(dataset, model, image, true_label, methods, raw_image, pil_preprocessor,
                                            tensor_preprocessor, BATCH_SIZE)
        ar.record_attribution(attributions)
        image_counter += 1
        pbar.update(1)
        if (image_counter+1) % PACKAGE_SIZE == 0:
            ar.save()
            ar = None


def get_max_noise_attributions(tasks):
    max_noise = 0
    for task in tasks:
        max_noise = max(max_noise, task["nr_noise"])
    return max_noise


def task_consistency_check(tasks, ensemble_techniques):
    titles = []
    for task in tasks:
        # Check for duplicate titles
        assert(not task["title"] in titles)
        titles.append(task["title"])

        assert(task["technique"] == "mean" or task["technique"] == "var" or task["technique"] == "rbm")
        assert(len(task["individual_methods"]) == len(ensemble_techniques))
        assert(task["nr_noise"] >= 0)
        assert(task["nr_noise"] is not None)
        if task["technique"] == "rbm":
            assert(task["rbm"] is not None)


def calculate_ensembles(title, ensemble_experiment_name, tasks, pos_filter):
    nr_noise_experiments = get_max_noise_attributions(tasks)
    files = os.listdir(AttrbutionsRecord.get_directory(title, "individual_attribution"))
    for file in tqdm(files):
        ar = AttrbutionsRecord.load(title, "individual_attribution", file)
        ar.start_ensemble(ensemble_experiment_name)
        ar.record_ensemble_tasks(tasks)
        task_consistency_check(tasks, ar.methods)

        for i in range(len(ar.attributions)):
            noise_attributions = compute_noise_attributions(ar.images[i], nr_noise_experiments)
            ar.record_noise_attributions(noise_attributions)

            ensemble_attributions = compute_ensembles(ar.attributions[i], ar.noise_attributions[i], tasks, pos_filter)
            ar.record_ensemble_attributions(ensemble_attributions)

        if pos_filter:
            ar.filter_negative_attributions()
        ar.save()


def calc_statistics_package(model, image, raw_image, attribution, label, pixel_group_size, irof_segments, irof_sigma,
                            irof, insert, delete):
    insert_statistic, delete_statistic, irof_statistic = (None, None, None)

    if insert:
        insert_statistic = pixel_score(True, model, image, attribution, label, pixel_group_size, BATCH_SIZE)
    if delete:
        delete_statistic = pixel_score(False, model, image, attribution, label, pixel_group_size, BATCH_SIZE)
    if irof:
        irof_statistic = irof_score(model, image, attribution, raw_image, label, irof_segments, irof_sigma, BATCH_SIZE)

    return insert_statistic, delete_statistic, irof_statistic


def calculate_statistics(title, ensemble_experiment_name, irof_score, insert_score,
                         delete_score, measure_individual_methods=True, measure_ensembles=True,
                         measure_flipped_rbms=True):
    if measure_flipped_rbms:
        assert(measure_ensembles)

    files = os.listdir(AttrbutionsRecord.get_directory(title, ensemble_experiment_name))
    model = None

    for file in tqdm(files):
        ar = AttrbutionsRecord.load(title, ensemble_experiment_name, file)
        if model is None:
            model = load_model(ar.dataset, ar.model).to(device)
        ar.create_statistics_table(measure_individual_methods, measure_ensembles, insert_score, delete_score,
                                   irof_score)

        if ar.dataset == "cifar10" or ar.dataset == "mnist":
            pixel_group_size = 1
            irof_segments = 60
            irof_sigma = 5
        elif ar.dataset == "imagenet":
            pixel_group_size = 20
            irof_segments = 200
            irof_sigma = 5

        for i in range(len(ar.attributions)):

            if measure_individual_methods:
                for j in range(len(ar.attributions[i])):
                    insert, delete, irof = calc_statistics_package(model, ar.images[i], ar.raw_images[i],
                                                                   ar.attributions[i][j], ar.true_labels[i],
                                                                   pixel_group_size, irof_segments, irof_sigma,
                                                                   irof_score, insert_score, delete_score)
                    ar.record_attributions_statistics(ar.methods[j], insert, delete, irof)

            if measure_ensembles:
                for j in range(len(ar.ensemble_attributions[i])):
                    attr_normalized = normalize_abs(ar.ensemble_attributions[i][j])
                    insert, delete, irof = calc_statistics_package(model, ar.images[i], ar.raw_images[i],
                                                                   attr_normalized, ar.true_labels[i], pixel_group_size,
                                                                   irof_segments, irof_sigma, irof_score, insert_score,
                                                                   delete_score)

                    insert_flip, delete_flip, irof_flip, prefer_flip = (None, None, None, False)
                    if measure_flipped_rbms:
                        if ar.ensemble_tasks[j]["technique"] == "rbm":
                            attr_normalized = normalize_abs(1-ar.ensemble_attributions[i][j])
                            insert_flip, delete_flip, irof_flip = calc_statistics_package(
                                model, ar.images[i], ar.raw_images[i], attr_normalized,
                                ar.true_labels[i], pixel_group_size, irof_segments, irof_sigma, irof_score,
                                insert_score, delete_score)

                            prefer_flip = True if insert_flip > insert else False
                    ar.record_ensemble_statistics(ar.ensembles[j], insert, delete, irof, insert_flip, delete_flip,
                                                  irof_flip, prefer_flip)
        ar.save()


def compute_score_table(title, ensemble_experiment_name):
    columns = None
    data = None

    files = os.listdir(AttrbutionsRecord.get_directory(title, ensemble_experiment_name))
    for file in tqdm(files):
        ar = AttrbutionsRecord.load(title, ensemble_experiment_name, file)

        if columns is None:
            columns = ["Method"]
            if ar.measure_insert:
                columns.append("Insert Mean")
                columns.append("Insert Std.")
            if ar.measure_delete:
                columns.append("Delete Mean")
                columns.append("Delete Std.")
            if ar.measure_irof:
                columns.append("IROF Mean")
                columns.append("IROF Std.")

            data = ar.statistics.copy()
            continue
        for key in data.keys():
            data[key] += ar.statistics[key]

    table_data = []
    for key in data.keys():
        data[key] = np.asarray(data[key])
        new_row = [key]

        for j in range(data[key].shape[1]):
            new_row.append(np.average(data[key][:, j]))
            new_row.append(np.std(data[key][:, j]))
        table_data.append(new_row)
    df = pd.DataFrame(table_data, columns=columns)
    return df





title = "mnist_test"
ensemble = "mean, rbm first test"

task1 = dict({
    "title": "mean",
    "technique": "mean",
    "individual_methods": [1, 1, 1, 1 ],
    "nr_noise": 0,
    "rbm": None,
})

task2 = dict({
    "title": "rbm",
    "technique": "mean",
    "individual_methods": [1, 1, 1, 1],
    "nr_noise": 0,
    "rbm": [BernoulliRBM(n_components=1, batch_size=10, learning_rate=0.01, n_iter=100)]
})

# generate_explanations(title, "mnist", ["DeepLift", "IntegratedGradient", "Saliency", "Gray"], 10)
# calculate_ensembles(title, ensemble, [task1, task2], False)
# # calculate_statistics(title, ensemble, True, True, True)
# df = compute_score_table(title, ensemble)



