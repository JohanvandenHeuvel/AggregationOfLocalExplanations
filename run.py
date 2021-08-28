from scripts.run_experiment import *
from scripts.attribution import *
from scripts.attribution_methods import attribution_method
import scripts.datasets as datasets
from models.model import get_model
import json

import torchvision
import torchvision.transforms as transforms

import os
import datetime

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
    "model": "Resnet18",
    "dataset": "small_imagenet",
    "attribution_methods": ["deeplift", "smoothgrad", "saliency"],
    "images": [39, 41],
}


###########################
#    running the code     #
###########################
def main():

    # transform = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # )
    # dataset = torchvision.datasets.CIFAR10(
    #     root="./data", train=False, download=True, transform=transform
    # )
    # dataloader = torch.utils.data.DataLoader(
    #     dataset, batch_size=4, shuffle=False, num_workers=2
    # )

    # classification model
    model = get_model(params["model"], device=device)

    # methods for explaining
    attribution_methods = [
        attribution_method(method, model) for method in params["attribution_methods"]
    ]

    # dataset and which images to explain the classification for
    dataset = datasets.get_dataset(params["dataset"], device)
    to_explain = dataset.get_image(params["images"])

    # for what label the image should be explained for
    labels = predict_label(model, to_explain)
    labels = labels.squeeze()

    # generate explanations
    attributions = generate_attributions(
        to_explain, labels, attribution_methods, device,
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


if __name__ == "__main__":
    write_params_to_disk()
    main()
