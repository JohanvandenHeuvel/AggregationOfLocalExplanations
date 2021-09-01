import torch

from scripts.pixel_manipulation import PixelManipulationBase


class PixelRelevancyDataset(PixelManipulationBase):
    def __init__(self, image, attribution, insert, batch_size, package_size, device):
        PixelManipulationBase.__init__(
            self, image, attribution, insert, batch_size, device
        )
        self._package_size = package_size
        self._device = device

        self.generate_pixel_batches()
        self.generate_initial_image()

        gauss_sum = int(self._batch_size * (self._batch_size + 1) / 2)
        self.generate_temp_baseline(gauss_sum * self._package_size)

    def generate_pixel_batches(self):
        # For simplicity: Ensure that all packages have the same size.
        max_nr_pixels = (
            self.width * self.height - self.width * self.height % self._package_size
        )
        # Sort pixels in descending order by attribution score
        pixel_relevancy_desc = torch.argsort(-self._attribution.flatten())[
            :max_nr_pixels
        ]
        # Add placeholder for original image
        placeholder = torch.LongTensor(self._package_size * [0]).to(self._device)
        pixel_relevancy_desc = torch.cat((pixel_relevancy_desc, placeholder))
        # Form groups of size package_size
        pixel_relevancy_groups = pixel_relevancy_desc.reshape(-1, self._package_size)
        # Forms batches of groups: (batch_size x package_size)
        self._pixel_batches = torch.split(
            pixel_relevancy_groups, self._batch_size, dim=0
        )

    def _gen_indices(self, index):
        batch_size = self._get_batch_size(index)

        # Create a matrix of indices of size [batch_size, package_size * batch_size]
        template_indices = self._pixel_batches[index].view(1, -1).repeat(batch_size, 1)
        # Shift each batch by total amount of pixels of previous image
        batch_indices = PixelManipulationBase._index_shift(
            template_indices.long(), self.width * self.height
        )

        # For each package only keep the previous pixels and package_size additional pixels
        keep_index_template = torch.cat(
            [torch.arange(0, self._package_size * i) for i in range(1, batch_size + 1)]
        )
        template_indices = template_indices.reshape(-1)[keep_index_template]
        template_indices = self._color_channel_shift(template_indices)

        # Do the same for the batches
        nr_pixel_in_group = self._package_size * batch_size
        keep_index_batch = torch.cat(
            [
                torch.arange(0, self._package_size * i) + (i - 1) * nr_pixel_in_group
                for i in range(1, batch_size + 1)
            ]
        )
        batch_indices = batch_indices.reshape(-1)[keep_index_batch]
        batch_indices = self._color_channel_shift(batch_indices)

        return template_indices, batch_indices

    def _get_fake_image_size(self):
        return self._package_size
