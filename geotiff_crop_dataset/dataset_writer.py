"""
Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-07-24
Description: A Pytorch Dataloader for tif image files that dynamically crops the image.
"""

import itertools

import rasterio

from geotiff_crop_dataset.dataset_reader import CropDatasetReader


class CropDatasetWriter:
    def __init__(self, img_path: str, crop_size: int, profile: dict):
        super().__init__()

        self.img_path = img_path
        self.crop_size = crop_size
        self.raster = rasterio.open(img_path, 'w', **profile)

        _y0s = range(0, self.raster.height, self.crop_size)
        _x0s = range(0, self.raster.width, self.crop_size)
        self.y0x0 = list(itertools.product(_y0s, _x0s))

    @classmethod
    def from_reader(cls, img_path: str, crop_size: int, reader: CropDatasetReader):
        """Create a CropDatasetWriter using a CropDatasetReader instance.
            Defines the geo-referencing, cropping, and size parameters using an existing raster image.
        
        Args:
            img_path: Path to the file you want to create. 
            crop_size: The size of the cropped section to be written. 
            reader: An instance of a CropDatasetReader from which to copy geo-referencing parameters. 

        Returns:
            CropDatasetWriter
        """
        self = cls(img_path, crop_size=crop_size, profile=reader.raster.profile)
        self.y0x0 = reader.y0x0

    def __setitem__(self, idx: int, value):
        y0, x0 = self.y0x0[idx]

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
