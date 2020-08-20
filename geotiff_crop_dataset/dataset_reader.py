"""
Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-07-24
Description: A Pytorch Dataloader for tif image files that dynamically crops the image.
"""

import itertools
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Union

import numpy as np
import rasterio


class Dataset(ABC):
    @abstractmethod
    def __len__(self):
        raise NotImplemented

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplemented


class CropDatasetReader(Dataset):
    def __init__(self, img_path: str, crop_size: int, padding: Optional[int] = 0, stride: Optional[int] = None,
                 fill_value: Optional[Union[int, float]] = None, transform: Optional[Callable] = None):
        """A Pytorch data loader that returns cropped segments of a tif image file.

        Args:
            img_path: The path to the image file to make the dataset from.
            crop_size: The desired edge length for each cropped section. Returned images will be square.
            padding: The amount of padding to add around each crop section from the adjacent image areas.
                Defaults to 0.
            stride: The stride length between cropped sections. Defaults to crop_size.
            fill_value: The value to fill in border regions of nodata areas of the image.
                Defaults to image nodata value.
            transform: Optional Pytorch style data transform to apply to each cropped section.
        """
        super().__init__()

        self.img_path = img_path
        self.crop_size = crop_size
        self.padding = padding
        self.stride = stride if stride is not None else crop_size

        self.raster = rasterio.open(img_path, 'r')

        if fill_value is not None:
            self.fill_value = fill_value
        elif hasattr(self.raster, "nodata") and self.raster.nodata:
            self.fill_value = self.raster.nodata
        else:
            self.fill_value = 0

        self.transform = transform

        _y0s = range(0, self.raster.height, self.stride)
        _x0s = range(0, self.raster.width, self.stride)
        self.y0x0 = list(itertools.product(_y0s, _x0s))

    @property
    def y0(self) -> List[int]:
        return [a[0] for a in self.y0x0]

    @property
    def x0(self) -> List[int]:
        return [a[1] for a in self.y0x0]

    def __len__(self) -> int:
        return len(self.y0x0)

    def __getitem__(self, idx: int) -> Any:
        y0, x0 = self.y0x0[idx]

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

    def close(self):
        self.raster.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
