import torch
from torch.utils.data import Dataset
import abc


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
        self._baseline = torch.mean(
            image.reshape(self.color_channels, self.width, self.height), dim=(1, 2)
        )

        self._temp_baseline = None
        self._temp_image = None

    @abc.abstractmethod
    def generate_pixel_batches(self):
        return

    def generate_initial_image(self):
        if self._insert:
            # Create a new image of original width & height with every pixel set to the baseline color
            self._temp_image = (
                self._baseline.view(self.color_channels, 1)
                .repeat(1, self.width * self.height)
                .flatten()
            )
        else:
            self._temp_image = self._image
        self._temp_image = self._temp_image.reshape(-1)

    def generate_temp_baseline(self, count):
        # Keep baseline image multiply times to avoid repeat the generation every iteration
        self._temp_baseline = (
            self._baseline.view(self.color_channels, 1).repeat(1, count).reshape(-1)
        )

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
        return torch.cat(
            [indices * self.color_channels + i for i in range(self.color_channels)]
        )

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
            template_indices = template_indices[
                : -self._get_fake_image_size() * self.color_channels * batch_size
            ]
            batch_indices = batch_indices[
                : -self._get_fake_image_size() * self.color_channels * batch_size
            ]

        # Modify the pixels
        if self._insert:
            image_batch[batch_indices] = self._image.flatten()[template_indices]
        else:
            image_batch[batch_indices] = self._temp_baseline[0 : len(template_indices)]

        # Reshape the image to proper sizes as required by the network
        image_batch = image_batch.reshape(
            -1, self.color_channels, self.width, self.height
        )

        if index == self.__len__() - 1:
            # Only in the last run: Add the original image
            image_batch[batch_size - 1] = self._image
        else:
            # Save last image for the next run
            self._temp_image = image_batch[batch_size - 1]

        return image_batch
