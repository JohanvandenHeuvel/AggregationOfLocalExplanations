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
            )
        else:
            self._temp_image = self._image
        self._temp_image = self._temp_image.flatten()

    def generate_temp_baseline(self):
        # Keep baseline image multiply times to avoid repeat the generation every iteration
        self._temp_baseline = (
            self._baseline.view(self.color_channels, 1).repeat(1, self.nr_pixels).flatten()
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

    def _index_shift(self, matrix, add_per_row):
        # Adds (i-1) * add_per_row to every cell in row i. Nothing for row 1
        factor = add_per_row * torch.diag(torch.arange(0, len(matrix), device=self._device)).float()
        new_matrix = factor @ torch.ones_like(matrix, device=self._device).float() + matrix
        return new_matrix.long()

    @abc.abstractmethod
    def _gen_indices(self, index):
        return

    @property
    def nr_pixels(self):
        return self.width * self.height

    def _color_channel_shift(self, indices):
        # For every color channel shift by nr_pixels
        return torch.stack(
            [indices + i*self.nr_pixels for i in range(self.color_channels)]
        ).to(self._device).T.flatten()

    def _batch_shift(self, indices, pixel_per_image):
        # Depending on pixel_per_image create an array of the following form:
        # [0 0 1 1 1 2 2 2 2 2 2]
        # How often a number is repeated depends on pixel_per_image
        nr_pixels_cum = torch.cumsum(pixel_per_image, 0)
        image_indices = [torch.Tensor(nr_pixels_cum[i]*[i]) for i in range(len(nr_pixels_cum))]
        image_indices = torch.cat(image_indices).to(self._device).long()

        # Multiply it with the total number of data points per image
        image_indices_shift = image_indices * self.color_channels * self.nr_pixels

        # Expand the shift for each color channel
        nr_man_pixels = int(len(indices) / self.color_channels)
        image_indices_shift = image_indices_shift.reshape(-1, 1).expand(nr_man_pixels, self.color_channels)
        image_indices_shift = image_indices_shift.flatten()

        # Shift the original indices
        batch_indices = indices + image_indices_shift

        return batch_indices

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
            image_batch[batch_indices] = self._temp_baseline[template_indices]

        # Reshape the image to proper sizes as required by the network
        image_batch = image_batch.reshape(
            -1, self.color_channels, self.width, self.height
        )

        if index == self.__len__() - 1:
            # Only in the last run: Add the original image
            image_batch[batch_size - 1] = self._image
        else:
            # Save last image for the next run
            self._temp_image = image_batch[-1]

        return image_batch
