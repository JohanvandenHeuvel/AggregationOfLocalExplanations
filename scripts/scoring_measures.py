import numpy as np
import torch
from skimage.segmentation import slic
import torch.nn.functional as F
from sklearn.metrics import auc

def irof_score(
    model, image, attribution, raw_image, label, n_segments, sigma, batch_size
):
    color_channels, width, height = image.shape
    # Baseline = mean per color channel
    base_color = torch.mean(image.view(image.shape[0], -1), dim=1).reshape(
        color_channels, -1
    )

    # Compute segments and mean relevancy per segment
    segments = slic(raw_image.numpy().transpose(1, 2, 0), n_segments, sigma).reshape(-1)
    segment_mean_relev = np.asarray(
        [np.mean(attribution.reshape(-1)[segments == seg]) for seg in range(n_segments)]
    )

    # Rank segments using attributions
    ranking = (-segment_mean_relev).argsort()  # Sort descending
    ranking = np.array_split(ranking, int(np.floor(len(ranking) / batch_size)))

    original_prob = calculate_probs(
        model, label, image.reshape(-1, image.shape[0], image.shape[1], image.shape[2])
    )[0]

    temp_image = image.clone().reshape(color_channels, -1)
    new_probs = np.zeros(len(segment_mean_relev))

    # Calculate new probs. batch-wise
    new_images = torch.zeros(
        (batch_size, color_channels, width * height), device=get_device()
    )
    for i, super_segment_block in enumerate(ranking):
        for j, seg in enumerate(super_segment_block):
            temp_image[:, segments == seg] = base_color.repeat(
                1, np.sum(segments == seg)
            )
            new_images[j] = temp_image

        new_images = new_images.reshape(-1, color_channels, width, height).double()
        new_probs[
            batch_size * i : batch_size * i + len(super_segment_block)
        ] = calculate_probs(model, label, new_images)
        new_images = new_images.reshape(-1, color_channels, width * height).double()

    y = new_probs / original_prob
    score = auc(np.arange(0, len(new_probs)), y) / len(new_probs)
    return score, original_prob, y


def calculate_probs(model, label, input):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    input.double()
    model.double()
    input.requires_grad = False
    model.eval()

    result = model(input)
    result = F.softmax(result, dim=1)
    result = result[:, label]
    result = result.cpu().detach().numpy()
    return result


def pixel_score(insert, model, image, attribution, label, pixel_group_size, batch_size):
    color_channels, width, height = image.shape
    max_pixels = width * height - width * height % pixel_group_size

    # Baseline = Mean color = Mean per color channel
    image = image.reshape(color_channels, width * height)
    baseline = torch.mean(image, dim=1)
    if len(baseline) == 0:
        baseline = baseline[0]

    original_prob = calculate_probs(
        model, label, image.detach().reshape(-1, color_channels, width, height)
    )[0]
    new_probs = np.zeros(int(max_pixels / pixel_group_size))

    # Get most important pixels first
    pixel_order = np.argsort(attribution.flatten())[-max_pixels:][::-1]
    pixel_order = torch.LongTensor(pixel_order.reshape(-1, pixel_group_size).copy())
    pixel_order = torch.split(pixel_order, batch_size, dim=0)

    if insert:
        temp_image = baseline.reshape(color_channels, 1).repeat(1, width * height)
    else:
        temp_image = image.clone().detach()
    image_batch = temp_image.unsqueeze(0).repeat(batch_size, 1, 1)

    for j, pixel_super_block in enumerate(pixel_order):
        for k, pixel_block in enumerate(pixel_super_block):
            if insert:
                temp_image[:, pixel_block] = image[:, pixel_block]
            else:
                temp_image[:, pixel_block] = baseline.reshape(color_channels, 1).repeat(
                    1, pixel_group_size
                )
            image_batch[k, :] = temp_image

        image_batch = image_batch.reshape(batch_size, color_channels, width, height)
        new_probs[
            j * batch_size : j * batch_size + len(pixel_super_block)
        ] = calculate_probs(model, label, image_batch)[0 : len(pixel_super_block)]
        image_batch = image_batch.reshape(batch_size, color_channels, width * height)

    y = new_probs / original_prob
    score = auc(np.arange(0, len(new_probs)), y) / len(new_probs)

    return score, original_prob, y
