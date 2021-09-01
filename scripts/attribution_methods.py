from captum.attr import *
import torch


def generate_attributions(image_batch, label_batch, model, methods, device="cpu"):

    size = [len(methods)] + [image_batch.shape[0], image_batch.shape[2], image_batch.shape[3]]
    a = torch.empty(size=size).to(device)
    image_batch.requires_grad = True

    for i, m in enumerate(methods):

        method = attribution_method(m, model)
        attr = method(image_batch, label_batch)
        # sum over the color channels
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


def noise_normal(**kwargs):
    def f(x, y):
        return torch.randn(x.shape).to(x.device)

    return f


def noise_uniform(**kwargs):
    def f(x, y):
        return torch.rand(x.shape).to(x.device)

    return f
