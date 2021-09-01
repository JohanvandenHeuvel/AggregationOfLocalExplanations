import torch
import torch.nn.functional as F


def calculate_probs(model, images):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    images.requires_grad = False
    model.eval()

    result = F.softmax(model(images), dim=1)
    return result


def predict_label(model, images):
    probs = calculate_probs(model, images)
    # take softmax over the classes
    output = F.softmax(probs, dim=1)
    # take the most likely class
    _, label = torch.topk(output, 1)
    return label