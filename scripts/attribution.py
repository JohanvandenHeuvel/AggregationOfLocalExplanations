import numpy as np

from captum.attr import IntegratedGradients
from captum.attr import DeepLift
from captum.attr import GuidedBackprop
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import Saliency
import torch
import torch.nn.functional as F
from lime import lime_image

from .normalize import normalize_sum1, normalize_abs

SMOOTHGRAD_SAMPLES = 100


def generate_attributions(image_batch, label_batch, methods, device="cpu"):

    attributions = []
    # for image_idx, (image, true_label) in tqdm(enumerate(dataloader)):
    #
    #     if image_idx > n_images:
    #         break

    # predicted_label = predict_label(model, image_batch)
    #
    # # only use attributions where the model predicts correctly
    # if predicted_label == true_label:
    #
    #     print("true")

    for i, m in enumerate(methods):
        attributions.append(m(image_batch, label_batch))

    return attributions


def compute_explanations(
    model, image, true_label, methods, preproc_pil, batch_size,
):
    attributions = [
        compute_single_explanation(
            model, m, image, true_label, preproc_pil, batch_size,
        )
        for m in methods
    ]

    return attributions


def compute_noise_attributions(image, nr_noise_explanations):
    if nr_noise_explanations > 0:
        noise_attributions = np.random.normal(
            size=(nr_noise_explanations, 1, image.size()[-2], image.size()[-1])
        )
        return noise_attributions
    else:
        return None


def compute_single_explanation(
    model, method, image, true_label, preproc_pil, batch_size,
):
    # TODO what is empty_cache() doing
    torch.cuda.empty_cache()

    # # TODO: Lime
    # if method == "Lime":
    #     # Lime does not work with greyscale images
    #     explainer = lime_image.LimeImageExplainer()
    #
    #     # TODO why are we not using the same image for the input of the model and the one we use for explaining?
    #     def batch_predict(images):
    #         model.eval()
    #         batch = torch.stack(tuple(preproc_tensor(i) for i in images), dim=0)
    #         logits = model(batch)
    #         probs = F.softmax(logits, dim=1)
    #         # TODO why detach here
    #         return probs.detach().cpu().numpy()
    #
    #     explanation = explainer.explain_instance(
    #         np.array(preproc_pil(raw_image)).reshape(1, 28, 28, 1),
    #         batch_predict,
    #         top_labels=5,
    #         hide_color=0,
    #         num_samples=20,
    #     )
    #
    #     segments = explanation.segments
    #     local_explanation = explanation.local_exp[true_label]
    #     for segment, value in local_explanation:
    #         segments[segments == segment] = value
    #
    #     attribution = segments.reshape(image.size()[2], image.size()[3])

    if "IntegratedGradient" in method:
        if method == "IntegratedGradient" or "black" in method:
            ig_baseline = np.zeros(image.shape[0:])
        elif "random" in method:
            ig_baseline = np.random.uniform(low=0, high=255, size=(image.shape))
        elif "white" in method:
            ig_baseline = 255 * np.ones(image.shape)
        else:
            raise ValueError(f"Integrated gradient method {method} not supported")

        ig_baseline = (
            preproc_tensor(ig_baseline.transpose(1, 2, 0)).double().unsqueeze(0)
        )
        image = image.unsqueeze(0)
        attribution = IntegratedGradients(model.double()).attribute(
            image,
            target=true_label,
            baselines=ig_baseline,
            internal_batch_size=batch_size,
        )
        # IG Baseline: 1x3x32x32
        # Image: 1x3x32x32
    elif method == "DeepLift":
        attribution = DeepLift(model).attribute(image, target=true_label)
    elif method == "Saliency":
        image = image.unsqueeze(0)
        attribution = Saliency(model).attribute(image, target=true_label)
    elif method == "Occlusion":
        image = image.unsqueeze(0)
        attribution = Occlusion(model).attribute(
            image, target=true_label, sliding_window_shapes=(1, 5, 5)
        )
    elif method == "GuidedBackprop":
        image = image.unsqueeze(0)
        attribution = GuidedBackprop(model).attribute(image, target=true_label)
    elif method == "SmoothGrad":
        image = image.unsqueeze(0)
        attribution = NoiseTunnel(Saliency(model)).attribute(
            image,
            target=true_label,
            nt_type="smoothgrad",
            nt_samples=SMOOTHGRAD_SAMPLES,
            # TODO this constant isn't defined
            nt_samples_batch_size=BATCH_SIZE_EXPLANATION,
        )
    elif method == "Gray":
        attribution = torch.mean(image, axis=0).reshape(
            1, 1, image.shape[1], image.shape[2]
        )
    else:
        raise ValueError

    # TODO why are we moving it to cpu here? Torch can compute mean as well
    # attribution = attribution.cpu().detach().numpy()
    # attribution = np.mean(attribution, axis=(0, 1))  # Remove color

    # remove color
    attribution = torch.mean(attribution, dim=(0, 1))

    return attribution
