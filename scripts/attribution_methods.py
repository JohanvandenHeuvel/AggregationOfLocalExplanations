import numpy as np
from captum.attr import *
import torch
from captum._utils.models.linear_model import SkLearnLinearRegression, SkLearnLasso
from captum.attr._core.lime import get_exp_kernel_similarity_function
from kornia.filters import gaussian_blur2d
from skimage.segmentation import slic

from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def generate_attributions(
    image_batch, label_batch, model, params, attribution_params, device="cpu"
):

    methods = params["attribution_methods"]

    size = [len(methods)] + [
        image_batch.shape[0],
        image_batch.shape[2],
        image_batch.shape[3],
    ]
    a = torch.empty(size=size).to(device)
    image_batch.requires_grad = True

    for i, m in enumerate(methods):
        attr_args = attribution_params[m]
        if "lime" in m or "gradientshap" in m:
            # TODO if doing in batches then gradientshap doesn't use one label per sample
            method = attribution_method(m, model, **attr_args)
            attr = torch.empty(size=image_batch.shape).to(device)
            for idx, img in enumerate(image_batch):
                attr[idx] = method(img, label_batch[idx])
        elif "integrated_gradients" in m:
            method = attribution_method(
                m, model, color_dim=image_batch.shape[1], **attr_args
            )
            attr = method(image_batch, label_batch)
        else:
            method = attribution_method(m, model, **attr_args)
            attr = method(image_batch, label_batch)
        # sum over the color channels
        if len(attr.shape) > 3:
            attr = torch.mean(attr, dim=1)
        a[i] = attr

    return a


def attribution_method(name, model, **kwargs):

    if "integrated_gradient" in name:
        return integrated_gradients(model, **kwargs)

    if "deeplift" in name:
        return deeplift(model, **kwargs)

    if "saliency" in name:
        return saliency(model, **kwargs)

    if "occlusion" in name:
        return occlusion(model, **kwargs)

    if "guidedbackprop" in name:
        return guidedbackprop(model, **kwargs)

    if "smoothgrad" in name:
        return smoothgrad(model, **kwargs)

    if "lime" in name:
        return lime(model, **kwargs)

    if "gradientshap" in name:
        return gradientshap(model, **kwargs)

    if "gray_image" in name:
        return gray_image(**kwargs)

    if "noise_normal" in name:
        return noise_normal(**kwargs)

    if "noise_uniform" in name:
        return noise_uniform(**kwargs)


def attribute_image_features(model, name, input, label, **kwargs):
    model.zero_grad()
    algorithm = attribution_method(name, model, **kwargs)
    tensor_attributions = algorithm(input, label)
    return tensor_attributions


def integrated_gradients(model, baseline, color_dim):

    if baseline == "blur":

        def f(x, y):
            x = gaussian_blur2d(x, (3, 3), (4, 4))
            return IntegratedGradients(model).attribute(x, target=y)

        return f

    else:
        if baseline == "black":
            color = torch.Tensor([0]).float().expand(color_dim).to(device)
        elif baseline == "random":
            color = torch.rand(color_dim).float().to(device)
        else:
            raise ValueError

        def f(x, y):
            baseline_color = color.reshape(1, -1, 1, 1)
            baselines = baseline_color.expand(
                x.shape[0], x.shape[1], x.shape[2], x.shape[3]
            )
            return IntegratedGradients(model).attribute(
                x, target=y, baselines=baselines
            )

        return f


def deeplift(model, **kwargs):
    def f(x, y):
        return DeepLift(model).attribute(x, target=y, **kwargs)

    return f


def saliency(model, **kwargs):
    f = lambda x, y: Saliency(model).attribute(x, target=y, **kwargs)
    return f


def occlusion(model, **kwargs):
    f = lambda x, y: Occlusion(model).attribute(
        x, target=y, sliding_window_shapes=(1, 5, 5), **kwargs
    )
    return f


def guidedbackprop(model, **kwargs):
    f = lambda x, y: GuidedBackprop(model).attribute(x, target=y, **kwargs)
    return f


def smoothgrad(model, **kwargs):
    nt = NoiseTunnel(Saliency(model))
    f = lambda x, y: nt.attribute(
        x, target=y, nt_type="smoothgrad", nt_samples=100, **kwargs
    )
    return f


def gray_image(**kwargs):
    def f(x, y):
        return x

    return f


def noise_normal(**kwargs):
    def f(x, y):
        return torch.randn(x.shape).to(x.device)

    return f


def noise_uniform(**kwargs):
    def f(x, y):
        return torch.rand(x.shape).to(x.device)

    return f


def lime(model, use_slic, n_slic_segments=100, n_samples=256, perturbations_per_eval=128):

    exp_eucl_distance = get_exp_kernel_similarity_function(
        "euclidean", kernel_width=1000
    )

    def f(x, y):
        lime_attr = Lime(
            model, SkLearnLinearRegression(), similarity_func=exp_eucl_distance,
        )

        # segment the image into superpixels
        if use_slic:
            img = x.permute(1, 2, 0).cpu().detach()
            seg = slic(
                img.to(torch.double).numpy(),
                n_segments=n_slic_segments,
                multichannel=True,
            )

            # to check if correctly transposed
            # print(img.shape)
            # print(seg.shape)
            # plt.imshow(mark_boundaries(img, seg))
            # plt.show()
            seg = torch.Tensor(seg).to(torch.long).to(device)

            return lime_attr.attribute(
                x.unsqueeze(0),
                target=y.unsqueeze(0),
                feature_mask=seg.unsqueeze(0),
                n_samples=n_samples,
                perturbations_per_eval=perturbations_per_eval,
            )

        else:
            return lime_attr.attribute(
                x.unsqueeze(0),
                target=y.unsqueeze(0),
                n_samples=n_samples,
                perturbations_per_eval=perturbations_per_eval,
            )

    return f


def gradientshap(model):
    def f(x, y):
        # Defining baseline distribution of images
        rand_img_dist = torch.cat([x.unsqueeze(0) * 0, x.unsqueeze(0) * 1])
        return GradientShap(model).attribute(
            x.unsqueeze(0), target=y.unsqueeze(0), baselines=rand_img_dist
        )

    return f
