from setuptools_scm import get_version

from geotiff_crop_dataset.dataset_reader import CropDatasetReader
from geotiff_crop_dataset.dataset_writer import CropDatasetWriter

__all__ = [CropDatasetReader, CropDatasetWriter]
__version__ = get_version(root='..', relative_to=__file__)
