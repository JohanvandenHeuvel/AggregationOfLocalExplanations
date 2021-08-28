from captum.attr import *


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


def attribute_image_features(model, name, input, label, **kwargs):
    model.zero_grad()
    algorithm = attribution_method(name, model, **kwargs)
    tensor_attributions = algorithm(input, label)
    return tensor_attributions


def deeplift(model, **kwargs):
    f = lambda x, y: DeepLift(model).attribute(x, target=y, **kwargs)
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
