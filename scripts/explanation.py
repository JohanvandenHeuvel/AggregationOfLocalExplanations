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

from normalize import normalize_sum1, normalize_abs

SMOOTHGRAD_SAMPLES = 100


def compute_single_explanation(dataset, net, method, image, label, raw_image, preproc_pil, preproc_tensor, batch_size):
    torch.cuda.empty_cache()
    net.eval()

    # TODO: Lime
    if method == "Lime":
        # Lime does not work with greyscale images
        explainer = lime_image.LimeImageExplainer()

        def batch_predict(images):
            net.eval()
            batch = torch.stack(tuple(preproc_tensor(i) for i in images),
                                dim=0)
            logits = net(batch)
            probs = F.softmax(logits, dim=1)
            return probs.detach().cpu().numpy()

        explanation = explainer.explain_instance(
            np.array(preproc_pil(raw_image)).reshape(1, 28, 28, 1),
            batch_predict,
            top_labels=5,
            hide_color=0,
            num_samples=20)

        segments = explanation.segments
        local_explanation = explanation.local_exp[label]
        for segment, value in local_explanation:
            segments[segments == segment] = value

        attribution = segments.reshape(image.size()[2], image.size()[3])
    else:
        if "IntegratedGradient" in method:
            if method == "IntegratedGradient" or "black" in method:
                ig_baseline = np.zeros(image.shape[0:])
            elif "random" in method:
                ig_baseline = np.random.uniform(low=0, high=255, size=(image.shape))
            elif "white" in method:
                ig_baseline = 255 * np.ones(image.shape)
            else:
                raise ValueError(f"Integrated gradient method {method} not supported")

            ig_baseline = preproc_tensor(ig_baseline.transpose(1,2,0)).double().unsqueeze(0)
            image = image.unsqueeze(0)
            attribution = IntegratedGradients(net.double()).attribute(image, target=label, baselines=ig_baseline,
                                                             internal_batch_size=batch_size)
            # IG Baseline: 1x3x32x32
            # Image: 1x3x32x32
        elif method == "DeepLift":
            image = image.unsqueeze(0)
            attribution = DeepLift(net).attribute(image, target=label)
        elif method == "Saliency":
            image = image.unsqueeze(0)
            attribution = Saliency(net).attribute(image, target=label)
        elif method == "Occlusion":
            image = image.unsqueeze(0)
            attribution = Occlusion(net).attribute(image, target=label, sliding_window_shapes=(1, 5, 5))
        elif method == "GuidedBackprop":
            image = image.unsqueeze(0)
            attribution = GuidedBackprop(net).attribute(image, target=label)
        elif method == "SmoothGrad":
            image = image.unsqueeze(0)
            attribution = NoiseTunnel(Saliency(net)).attribute(image, target=label, nt_type="smoothgrad",
                                                               nt_samples=SMOOTHGRAD_SAMPLES,
                                                               nt_samples_batch_size=BATCH_SIZE_EXPLANATION)
        elif method == "Gray":
            attribution = torch.mean(image, axis=0).reshape(1, 1, image.shape[1], image.shape[2])
        else:
            raise ValueError

        attribution = attribution.cpu().detach().numpy()
        attribution = np.mean(attribution, axis=(0, 1))  # Remove color
    return attribution


def compute_explanations(dataset, net, image, label, methods, raw_image, preproc_pil, preproc_tensor, batch_size):
    attributions = np.asarray([compute_single_explanation(dataset, net, m, image, label, raw_image, preproc_pil,
                                                          preproc_tensor, batch_size) for m in methods])

    return attributions


def compute_noise_attributions(image, nr_noise_explanations):
    if nr_noise_explanations > 0:
        noise_attributions = np.random.normal(size=(nr_noise_explanations, 1, image.size()[-2], image.size()[-1]))
        return noise_attributions
    else:
        return None

