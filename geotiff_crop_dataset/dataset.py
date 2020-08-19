"""
Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-07-24
Description: A Pytorch Dataloader for tif image files that dynamically crops the image.
"""

import itertools
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Union

import numpy as np
import rasterio


class Dataset(ABC):
    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError


class CropDatasetReader(Dataset):
    def __init__(self, img_path: str, crop_size: int, padding: Optional[int] = 0, stride: Optional[int] = None,
                 fill_value: Optional[Union[int, float]] = None, transform: Optional[Callable] = None):
        """A Pytorch data loader that returns cropped segments of a tif image file.

        :param img_path: str
            The path to the image file to make the dataset from.
        :param crop_size: int
            The desired edge length for each cropped section. Returned images will be square.
        :param padding: Optional[int]
            The amount of padding to add around each crop section from the adjacent image areas. Defaults to 0.
        :param stride: Optional[int]
            The stride length between cropped sections. Defaults to crop_size
        :param fill_value: Optional[Union[int, float]]
            The value to fill in border regions of nodata areas of the image. Defaults to image nodata value.
        :param transform: Optional[Callable]
            Optional Pytorch style data transform to apply to each cropped section.
        """
        super().__init__()

        self.img_path = img_path
        self.crop_size = crop_size
        self.padding = padding
        self.stride = stride if stride is not None else crop_size

        self.raster = rasterio.open(img_path, 'r')

        if fill_value is not None:
            self.fill_value = fill_value
        elif hasattr(self.raster, "nodata"):
            self.fill_value = self.raster.nodata
        else:
            self.fill_value = 0

        self.transform = transform

        _y0s = range(0, self.raster.height, self.stride)
        _x0s = range(0, self.raster.width, self.stride)
        self._y0x0s = list(itertools.product(_y0s, _x0s))

    @property
    def y0x0(self):
        return self._y0x0s

    @property
    def y0(self):
        return [a[0] for a in self._y0x0s]

    @property
    def x0(self):
        return [a[1] for a in self._y0x0s]

    def __len__(self) -> int:
        return len(self._y0x0s)

    def __getitem__(self, idx: int) -> Any:
        y0, x0 = self._y0x0s[idx]

        # Read the image section
        window = ((y0 - self.padding, y0 + self.crop_size + self.padding),
                  (x0 - self.padding, x0 + self.crop_size + self.padding))
        crop = self.raster.read(window=window, masked=True, boundless=True, fill_value=self.fill_value)

        # Fill nodata values
        crop = crop.filled(self.fill_value)

        if len(crop.shape) == 3:
            crop = np.moveaxis(crop, 0, 2)  # (c, h, w) => (h, w, c)
            if crop.shape[2] == 1:
                crop = np.squeeze(crop, axis=2)  # (h, w, c) => (h, w)

        if self.transform:
            crop = self.transform(crop)

        return crop


class CropDatasetWriter:
    def __init__(self, img_path: str, crop_size: int, profile: dict):
        super().__init__()

        self.img_path = img_path
        self.crop_size = crop_size
        self.raster = rasterio.open(img_path, 'w', **profile)

        _y0s = range(0, self.raster.height, self.crop_size)
        _x0s = range(0, self.raster.width, self.crop_size)
        self._y0x0s = list(itertools.product(_y0s, _x0s))

    @classmethod
    def from_reader(cls, img_path: str, crop_size: int, r: CropDatasetReader):
        """Create a CropDatasetWriter using a CropDatasetReader instance to define the geo-referencing, cropping, and
            size parameters.

        :param img_path: str
            Path to the file you want to create.
        :param crop_size: int
            The size of the cropped section to be written.
        :param r: CropDatasetReader
            An instance of a CropDatasetReader from which to copy geo-referencing parameters.
        :return:
            CropDatasetWriter instance
        """
        return cls(img_path, crop_size=crop_size, profile=r.raster.profile)

    def __setitem__(self, idx: int, value):
        y0, x0 = self._y0x0s[idx]

        # Read the image section
        window = ((y0, min(y0 + self.crop_size, self.raster.height)),
                  (x0, min(x0 + self.crop_size, self.raster.width)))

        self.raster.write(value[:, :self.raster.height, :self.raster.width], window=window)

    def close(self):
        self.raster.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
