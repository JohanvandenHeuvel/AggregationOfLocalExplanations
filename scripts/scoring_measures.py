import numpy as np
import abc
import torch
from skimage.segmentation import slic
from sklearn.metrics import auc
from torch.utils.data import Dataset
from numpy.lib import recfunctions as rfn

from models.predict import calculate_probs

# TODO: I still have the broad PyCharm setting.

def calc_scores(model, image_batch, label_batch, attributions, attr_titles, scores, device, **kwargs):
    for i, (image, label) in enumerate(zip(image_batch, label_batch)):
        for scoring_method in scores.keys():
            for attr, title in zip(attributions[:,i], attr_titles):
                score = calc_score(model, scoring_method, image, label, attr, device, **kwargs)
                scores[scoring_method][title].append(score)


def calc_score(model, scoring_method, image, label, attr, device, **kwargs):
    batch_size = kwargs.pop("batch_size", 10)
    package_size = kwargs.pop("package_size", 1)
    irof_segments = kwargs.pop("irof_segments", 60)
    irof_sigma = kwargs.pop("irof_sigma", 5)

    if scoring_method == "insert":
        dataset = PixelRelevancyDataset(image, attr, True, batch_size, package_size, device)
    elif scoring_method == "delete":
        dataset = PixelRelevancyDataset(image, attr, False, batch_size, package_size, device)
    elif scoring_method == "irof":
        dataset = IrofDataset(image, attr, batch_size, irof_segments, irof_sigma, device)
    else:
        raise ValueError

    probs = []
    for img_batch in dataset:
        probs += [calculate_probs(model, img_batch)[:, label]]

    probs = torch.cat(probs)
    rel_probs = probs / probs[-1]

    x = np.arange(0, len(rel_probs))
    y = rel_probs.detach().cpu().numpy()
    score = auc(x, y) / len(rel_probs)

    return score


class PixelManipulationBase(Dataset):
    """
    Requires that self._pixel_batches is defined in the constructor
    """
    def __init__(self, image, attribution, insert, batch_size, device):
        self._image = image
        self._batch_size = batch_size
        self._insert = insert
        self._attribution = attribution
        self._device = device

        self._pixel_batches = None  # Expected to be set by child class
        self._temp_image = None  # Expected to be set by child class

        # Baseline = Mean color
        self._baseline = torch.mean(image.reshape(self.color_channels, self.width, self.height), dim=(1, 2))

        self._temp_baseline = None
        self._temp_image = None

    @abc.abstractmethod
    def generate_pixel_batches(self):
        return

    def generate_initial_image(self):
        if self._insert:
            # Create a new image of original width & height with every pixel set to the baseline color
            self._temp_image = self._baseline.view(self.color_channels, 1).repeat(1, self.width * self.height).flatten()
        else:
            self._temp_image = self._image
        self._temp_image = self._temp_image.reshape(-1)

    def generate_temp_baseline(self, count):
        # Keep baseline image multiply times to avoid repeat the generation every iteration
        self._temp_baseline = self._baseline.view(self.color_channels, 1).repeat(1, count).reshape(-1)

    def __len__(self):
        return len(self._pixel_batches)

    @property
    def color_channels(self):
        return self._image.shape[0]

    @property
    def width(self):
        return self._image.shape[1]

    @property
    def height(self):
        return self._image.shape[2]

    def _get_batch_size(self, index):
        return len(self._pixel_batches[index])

    @staticmethod
    def _index_shift(matrix, add_per_row):
        # Adds (i-1) * add_per_row to every cell in row i. Nothing for row 1
        factor = add_per_row * torch.diag(torch.arange(0, len(matrix)))
        new_matrix = factor @ torch.ones_like(matrix) + matrix
        return new_matrix

    @abc.abstractmethod
    def _gen_indices(self, index):
        return

    def _color_channel_shift(self, indices):
        return torch.cat([indices * self.color_channels + i for i in range(self.color_channels)])

    @abc.abstractmethod
    def _get_fake_image_size(self):
        return

    def __getitem__(self, index):
        """
        Returns batch_size of images, where the most important pixels have been removed / added
        Important: Call the method with consecutive index values!
        """
        batch_size = self._get_batch_size(index)

        # Start with the image of the last run
        # Create batch of image [batch_size, color_channels x width x height]
        image_batch = self._temp_image.view(1, -1).repeat(batch_size, 1).flatten()

        # Get the indices that need to be modified from the image of the last run
        # template_indices = not unique, batch_indices = unique
        template_indices, batch_indices = self._gen_indices(index)

        if index == self.__len__() - 1:
            # Only in the last run: Ensure that there is no problem for the original image
            # Therefore remove the fake indices as added in the constructor
            template_indices = template_indices[:-self._get_fake_image_size() * self.color_channels * batch_size]
            batch_indices = batch_indices[:-self._get_fake_image_size() * self.color_channels * batch_size]

        # Modify the pixels
        if self._insert:
            image_batch[batch_indices] = self._image.flatten()[template_indices]
        else:
            image_batch[batch_indices] = self._temp_baseline[0:len(template_indices)]

        # Reshape the image to proper sizes as required by the network
        image_batch = image_batch.reshape(-1, self.color_channels, self.width, self.height)

        if index == self.__len__() - 1:
            # Only in the last run: Add the original image
            image_batch[batch_size - 1] = self._image
        else:
            # Save last image for the next run
            self._temp_image = image_batch[batch_size - 1]

        return image_batch


class PixelRelevancyDataset(PixelManipulationBase):
    def __init__(self, image, attribution, insert, batch_size, package_size, device):
        PixelManipulationBase.__init__(self, image, attribution, insert, batch_size, device)
        self._package_size = package_size
        self._device = device

        self.generate_pixel_batches()
        self.generate_initial_image()

        gauss_sum = int(self._batch_size * (self._batch_size + 1) / 2)
        self.generate_temp_baseline(gauss_sum * self._package_size)

    def generate_pixel_batches(self):
        # For simplicity: Ensure that all packages have the same size.
        max_nr_pixels = self.width * self.height - self.width * self.height % self._package_size
        # Sort pixels in descending order by attribution score
        pixel_relevancy_desc = torch.argsort(-self._attribution.flatten())[:max_nr_pixels]
        # Add placeholder for original image
        placeholder = torch.LongTensor(self._package_size*[0]).to(self._device)
        pixel_relevancy_desc = torch.cat((pixel_relevancy_desc, placeholder))
        # Form groups of size package_size
        pixel_relevancy_groups = pixel_relevancy_desc.reshape(-1, self._package_size)
        # Forms batches of groups: (batch_size x package_size)
        self._pixel_batches = torch.split(pixel_relevancy_groups, self._batch_size, dim=0)

    def _gen_indices(self, index):
        batch_size = self._get_batch_size(index)

        # Create a matrix of indices of size [batch_size, package_size * batch_size]
        template_indices = self._pixel_batches[index].view(1, -1).repeat(batch_size, 1)
        # Shift each batch by total amount of pixels of previous image
        batch_indices = PixelManipulationBase._index_shift(template_indices.long(), self.width * self.height)

        # For each package only keep the previous pixels and package_size additional pixels
        keep_index_template = torch.cat([torch.arange(0, self._package_size * i) for i in range(1, batch_size+1)])
        template_indices = template_indices.reshape(-1)[keep_index_template]
        template_indices = self._color_channel_shift(template_indices)

        # Do the same for the batches
        nr_pixel_in_group = self._package_size * batch_size
        keep_index_batch = torch.cat([torch.arange(0, self._package_size * i) + (i-1) * nr_pixel_in_group
                                      for i in range(1, batch_size+1)])
        batch_indices = batch_indices.reshape(-1)[keep_index_batch]
        batch_indices = self._color_channel_shift(batch_indices)

        return template_indices, batch_indices

    def _get_fake_image_size(self):
        return self._package_size

# TODO: Some parts of IrofDataset and PixelRelevancyDataset may be redundant, because they are optimized.
# It maybe be possible to merge some parts, but I would focus on other things now
class IrofDataset(PixelManipulationBase):
    def __init__(self, image, attribution, batch_size, irof_segments, irof_sigma, device):
        PixelManipulationBase.__init__(self, image,  attribution, False, batch_size, device)
        self._irof_segments = irof_segments
        self._irof_sigma = irof_sigma

        self._max_seg_pixels = None

        self.generate_pixel_batches()
        self.generate_initial_image()

        gauss_sum = int(self._batch_size * (self._batch_size + 1) / 2)
        self.generate_temp_baseline(gauss_sum * self._max_seg_pixels)

    def generate_pixel_batches(self):
        # Apply Slic algorithm to get superpixel areas
        img_np = self._image.detach().cpu().numpy().transpose(1, 2, 0)
        segments = slic(img_np, self._irof_segments, self._irof_segments).reshape(-1)
        nr_segments = np.max(segments)+1
        segments = torch.LongTensor(segments).to(self._device)

        # Attribution score of each segment = Mean attribution
        attr = self._attribution.reshape(-1)
        seg_mean = [torch.mean(attr[segments == seg]).item() for seg in range(nr_segments)]
        seg_mean = torch.FloatTensor(seg_mean).to(self._device)

        # Sort segments descending by mean attribution
        seg_rank = torch.argsort(-seg_mean)

        # Create lists of "shape" [batch_size, segment_size]
        # containing the indices of the segments
        self._pixel_batches = [[]]
        for seg in seg_rank:
            indices = (segments == seg).nonzero().flatten()
            IrofDataset._add_to_hierarhical_list(self._pixel_batches, self._batch_size, indices)

        # Add placeholder for original image
        IrofDataset._add_to_hierarhical_list(self._pixel_batches, self._batch_size, torch.Tensor([0]).to(self._device))

        # Save the statistics for generate_temp_baseline
        seg_count = [len(attr[segments == seg]) for seg in range(nr_segments)]
        self._max_seg_pixels = np.max(seg_count)

    @staticmethod
    def _add_to_hierarhical_list(list_element, target_size, item):
        if len(list_element[-1]) == target_size:
            list_element.append([])
        list_element[-1].append(item)

    def _gen_indices(self, index):
        batch_size = self._get_batch_size(index)

        # Get all pixels
        all_pixels = torch.cat(self._pixel_batches[index])

        # Create a matrix of indices of size [batch_size, all_pixels]
        template_indices = all_pixels.view(1, -1).repeat(batch_size, 1)
        # Shift each batch by total amount of pixels of previous image
        batch_indices = PixelManipulationBase._index_shift(template_indices.long(), self.width * self.height)

        # For each package only keep the previous pixels and package_size additional pixels
        lengths = torch.LongTensor([len(package) for package in self._pixel_batches[index]]).to(self._device)
        cumsum = torch.cumsum(lengths, dim=0)
        keep_index_template = torch.cat([torch.arange(0, s.item()) for s in cumsum]).to(self._device)
        template_indices = template_indices.reshape(-1)[keep_index_template]
        template_indices = self._color_channel_shift(template_indices)

        # Do the same for the batches
        nr_pixel_in_group = len(all_pixels)
        keep_index_batch = torch.cat([torch.arange(0, s.item()) + i*nr_pixel_in_group for i, s in enumerate(cumsum)])
        batch_indices = batch_indices.reshape(-1)[keep_index_batch]
        batch_indices = self._color_channel_shift(batch_indices)

        return template_indices, batch_indices

    def _get_fake_image_size(self):
        return 1
