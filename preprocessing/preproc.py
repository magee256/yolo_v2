import numpy as np
from skimage.transform import rescale


class ScaleImages:
    def __init__(self, target_width):
        """
        Prep image scaling target

        :type target_width: int
        """
        self.target_width = target_width

    def transform(self, img):
        """
        Scale image so first dimension is self.target_width
        for input to ImageNet
        """
        height = img.shape[0]
        return rescale(img, self.target_width / height, mode='constant')

    def fit(self, x, y):
        pass


class CropImages:
    def __init__(self, target_width):
        """
        Prep image scaling target

        :type target_width: int
        """
        self.target_width = target_width

    def transform(self, img):
        """
        Crop image until second dimension is at most length target_width

        Modification spread evenly across both sides.
        """
        pix_to_crop = img.shape[1] - self.target_width
        if pix_to_crop > 0:
            img = img[:, pix_to_crop // 2:-((pix_to_crop + 1) // 2), :]
        return img

    def fit(self, x, y):
        pass


class PadImages:
    def __init__(self, target_width):
        """
        Prep image scaling target

        :type target_width: int
        """
        self.target_width = target_width

    def transform(self, img):
        """
        Pad second dimension of images with black until target_width

        Modification spread evenly across both sides.
        """
        pix_to_pad = self.target_width - img.shape[1]
        if pix_to_pad > 0:
            img = np.pad(img, [(0, 0),
                               (pix_to_pad // 2, (pix_to_pad + 1) // 2),
                               (0, 0)], mode='constant')
        return img

    def fit(self, x, y):
        pass
