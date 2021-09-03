from captum.attr import *
import torch

from captum._utils.models.linear_model import SkLearnLinearRegression, SkLearnLasso
from captum.attr._core.lime import get_exp_kernel_similarity_function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_attributions(image_batch, label_batch, model, params, device="cpu"):

    methods = params["attribution_methods"]

    size = [len(methods)] + [
        image_batch.shape[0],
        image_batch.shape[2],
        image_batch.shape[3],
    ]
    a = torch.empty(size=size).to(device)
    image_batch.requires_grad = True

    for i, m in enumerate(methods):

        if m == "lime" or m == "gradientshap":
            # TODO if doing in batches then gradientshap doesn't use one label per sample
            method = attribution_method(m, model)
            attr = torch.empty(size=image_batch.shape).to(device)
            for idx, img in enumerate(image_batch):
                foo = method(img, label_batch[idx])
                attr[idx] = foo
        else:
            method = attribution_method(m, model)
            attr = method(image_batch, label_batch)
        # sum over the color channels
        if len(attr.shape) > 3:
            attr = torch.mean(attr, dim=1)
        a[i] = attr

    return a


def attribution_method(name, model, **kwargs):
    if name == "deeplift":
        return deeplift(model, **kwargs)

    if name == "saliency":
        return saliency(model, **kwargs)

    if name == "occlusion":
        return occlusion(model, **kwargs)

    if name == "guidedbackprop":
        return guidedbackprop(model, **kwargs)

    if name == "smoothgrad":
        return smoothgrad(model, **kwargs)

    if name == "lime":
        return lime(model, **kwargs)

    if name == "gradientshap":
        return gradientshap(model, **kwargs)

    if name == "gray_image":
        return gray_image(**kwargs)

    if name == "noise_normal":
        return noise_normal(**kwargs)

    if name == "noise_uniform":
        return noise_uniform(**kwargs)


def attribute_image_features(model, name, input, label, **kwargs):
    model.zero_grad()
    algorithm = attribution_method(name, model, **kwargs)
    tensor_attributions = algorithm(input, label)
    return tensor_attributions


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
        return torch.mean(x, dim=1).unsqueeze(1)

    return f


def noise_normal(**kwargs):
    def f(x, y):
        return torch.randn(x.shape).to(x.device)

    return f


def noise_uniform(**kwargs):
    def f(x, y):
        return torch.rand(x.shape).to(x.device)

    return f


def lime(model):

    exp_eucl_distance = get_exp_kernel_similarity_function(
        "euclidean", kernel_width=1000
    )

    def f(x, y):
        lime_attr = Lime(
            model, SkLearnLinearRegression(), similarity_func=exp_eucl_distance,
        )
        return lime_attr.attribute(
            x.unsqueeze(0),
            target=y.unsqueeze(0),
            n_samples=256,
            perturbations_per_eval=128,
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
