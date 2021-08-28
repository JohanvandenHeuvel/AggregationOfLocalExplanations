import numpy as np
import pickle
from os import path
from pathlib import Path
import torch

EXPERIMENTS_PATH = "data"


class AttrbutionsRecord:
    def __init__(self, title, dataset, package_id, methods, model):
        self.title = title
        self.dataset = dataset
        self.package_id = package_id
        self.methods = methods
        self.model = model
        self.rbm_configuration = None
        self.experiment_name = "individual_attribution"

        self.ensemble_tasks = None
        self.image_ids = []
        self.true_labels = []
        self.attributions = []
        self.noise_attributions = []
        self.ensemble_attributions = []

        self.measure_insert = None
        self.measure_delete = None
        self.measure_irof = None

        self.attributions_stat_irof = []
        self.attributions_stat_insert = []
        self.attributions_stat_delete = []

        self.ensemble_stat_irof = []
        self.ensemble_stat_insert = []
        self.ensemble_stat_delete = []

        self.rbm_flip_stat_irof = []
        self.rbm_flip_stat_insert = []
        self.rbm_flip_stat_delete = []

        self.images = []
        self.raw_images = []

        self.statistics = None

        if path.exists(
            AttrbutionsRecord.get_path(
                self.title, self.experiment_name, self.package_id
            )
        ):
            raise Exception("Experiment already exists")

    def get_ensemble_double_rbm(self, image_nr):
        attributions = []
        ensembles = []
        for i, attr in enumerate(self.ensemble_attributions[image_nr]):
            ensembles.append(self.ensembles[i])
            attributions.append(attr)
            if self.ensemble_tasks[i]["technique"] == "rbm":
                attributions.append(1 - attr)
                ensembles.append(self.ensembles[i] + " flipped")

        return_ = [(attributions[i], ensembles[i]) for i in range(len(ensembles))]
        return return_

        return attributions, ensembles

    def record_attribution(self, attributions):
        self.attributions.append(np.stack(attributions))

    def filter_negative_attributions(self):
        self.attributions = np.asarray(self.attributions)
        self.attributions[self.attributions < 0] = 0

    def record_noise_attributions(self, noise_attribution):
        if noise_attribution is None:
            self.noise_attributions.append(None)
        else:
            self.noise_attributions.append(attributions)

    def record_ensemble_attributions(self, ensemble_attribution):
        self.ensemble_attributions.append(ensemble_attribution)

    def record_image(self, image_id, label, image, raw_image):
        self.image_ids.append(image_id)
        self.true_labels.append(label)
        self.images.append(image)
        self.raw_images.append(raw_image)

    def record_ensemble_tasks(self, ensemble_tasks):
        self.ensemble_tasks = ensemble_tasks

    def create_statistics_table(
        self,
        measure_individual_methods,
        measure_ensembles,
        measure_insert,
        measure_delete,
        measure_irof,
    ):
        self.measure_insert = measure_insert
        self.measure_delete = measure_delete
        self.measure_irof = measure_irof

        self.statistics = dict()
        if measure_individual_methods:
            for m in self.methods:
                self.statistics[m] = []
        if measure_ensembles:
            for e in self.ensembles:
                self.statistics[e] = []

    @property
    def ensembles(self):
        if self.ensemble_tasks is None:
            return None
        ensembles = []
        for task in self.ensemble_tasks:
            ensembles.append(task["title"])
        return ensembles

    @classmethod
    def get_directory(cls, title, experiment_name):
        return f"{EXPERIMENTS_PATH}/{title}/{experiment_name}"

    @classmethod
    def get_path(cls, title, experiment_name, package_id):
        return f"{AttrbutionsRecord.get_directory(title, experiment_name)}/{package_id}.pkl"

    def save(self):
        Path(AttrbutionsRecord.get_directory(self.title, self.experiment_name)).mkdir(
            parents=True, exist_ok=True
        )

        for i in range(len(self.images)):
            self.images[i] = self.images[i].cpu().numpy()
            self.raw_images[i] = self.raw_images[i].cpu().numpy()

        with open(
            AttrbutionsRecord.get_path(
                self.title, self.experiment_name, self.package_id
            ),
            "wb",
        ) as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def start_ensemble(self, ensemble_experiment_name):
        assert self.experiment_name == "individual_attribution"
        self.experiment_name = ensemble_experiment_name

    def record_attributions_statistics(
        self, method, insert_statistic, delete_statistic, irof_statistic
    ):
        self.statistics[method].append(
            [insert_statistic[0], delete_statistic[0], irof_statistic[0]]
        )
        self.attributions_stat_insert.append(insert_statistic)
        self.attributions_stat_delete.append(delete_statistic)
        self.attributions_stat_irof.append(irof_statistic)

    def record_ensemble_statistics(
        self,
        ensemble,
        insert_statistic,
        delete_statistic,
        irof_statistic,
        insert_flip_statistic,
        delete_flip_statistic,
        irof_flip_statistic,
        prefer_flip,
    ):
        stats = []

        if prefer_flip:
            if self.measure_insert:
                stats.append(insert_flip_statistic[0])
            if self.measure_delete:
                stats.append(delete_flip_statistic[0])
            if self.measure_irof:
                stats.append(irof_flip_statistic[0])
        else:
            if self.measure_insert:
                stats.append(insert_statistic[0])
            if self.measure_delete:
                stats.append(delete_statistic[0])
            if self.measure_irof:
                stats.append(irof_statistic[0])
        self.statistics[ensemble].append(stats)

        self.ensemble_stat_insert.append(insert_statistic)
        self.ensemble_stat_delete.append(delete_statistic)
        self.ensemble_stat_irof.append(irof_statistic)

        self.rbm_flip_stat_insert.append(insert_flip_statistic)
        self.rbm_flip_stat_delete.append(delete_flip_statistic)
        self.rbm_flip_stat_irof.append(irof_flip_statistic)

    def no_flip_statistic(self):
        self.rbm_flip_statistics.append(None)

    @property
    def statistics_categories(self):
        attributions_titles = []
        ensembles_titles = []
        flips = []

        if (
            len(self.attributions_stat_irof) > 0
            or len(self.attributions_stat_insert) > 0
            or len(self.attributions_stat_delete) > 0
        ):
            attributions_titles = self.methods

        if (
            len(self.ensemble_stat_irof) > 0
            or len(self.ensemble_stat_insert) > 0
            or len(self.ensemble_stat_delete) > 0
        ):
            ensembles_titles = self.ensembles

        if len(self.rbm_flip_statistics) > 0:
            for i in range(len(self.ensembles)):
                if self.rbm_flip_statistics[i] is not None:
                    flips.append(True)
                else:
                    flips.append(False)

        return attributions_titles, ensembles_titles, flips

    @classmethod
    def load(cls, title, experiment_name, filename):
        directory = AttrbutionsRecord.get_directory(title, experiment_name)
        with open(f"{directory}/{filename}", "rb") as input:
            data = pickle.load(input)

        for i in range(len(data.images)):
            data.images[i] = torch.DoubleTensor(data.images[i])
            data.raw_images[i] = torch.DoubleTensor(data.raw_images[i])

        return data
