import numpy as np
import rasterio
import torch
from torchvision import transforms

from geotiff_crop_dataset.dataset import CropDataset


def _create_simple_test_img(tmpdir):
    simple_img = np.array([[1, 2], [3, 4]])

    p = str(tmpdir.mkdir("simple").join("simple_img.tif"))
    with rasterio.open(
            p,
            'w',
            driver='GTiff',
            height=simple_img.shape[0],
            width=simple_img.shape[1],
            count=1,
            dtype=rasterio.uint8) as dst:
        dst.write(simple_img.astype(rasterio.uint8), 1)

    return p


def test_simple_crop(tmpdir):
    p = _create_simple_test_img(tmpdir)

    ds = CropDataset(p, crop_size=1)
    assert ds[0] == np.array([[1]])
    assert ds[1] == np.array([[2]])
    assert ds[2] == np.array([[3]])
    assert ds[3] == np.array([[4]])

    ds = CropDataset(p, crop_size=2)
    assert np.all(ds[0] == np.array([[1, 2], [3, 4]]))


def test_padding(tmpdir):
    p = _create_simple_test_img(tmpdir)

    ds = CropDataset(p, crop_size=1, padding=1, fill_value=0)
    assert np.all(ds[0] == np.array([[0, 0, 0], [0, 1, 2], [0, 3, 4]]))
    assert np.all(ds[1] == np.array([[0, 0, 0], [1, 2, 0], [3, 4, 0]]))
    assert np.all(ds[2] == np.array([[0, 1, 2], [0, 3, 4], [0, 0, 0]]))
    assert np.all(ds[3] == np.array([[1, 2, 0], [3, 4, 0], [0, 0, 0]]))

    ds = CropDataset(p, crop_size=2, padding=1, fill_value=0)
    assert np.all(ds[0] == np.array([[0, 0, 0, 0],
                                     [0, 1, 2, 0],
                                     [0, 3, 4, 0],
                                     [0, 0, 0, 0]
                                     ]))

    ds = CropDataset(p, crop_size=2, padding=2, fill_value=8)
    assert np.all(ds[0] == np.array([
        [8, 8, 8, 8, 8, 8],
        [8, 8, 8, 8, 8, 8],
        [8, 8, 1, 2, 8, 8],
        [8, 8, 3, 4, 8, 8],
        [8, 8, 8, 8, 8, 8],
        [8, 8, 8, 8, 8, 8],
    ]))


def test_stride(tmpdir):
    p = _create_simple_test_img(tmpdir)

    ds = CropDataset(p, crop_size=2, stride=1, padding=2, fill_value=0)
    assert np.all(ds[0] == np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 2, 0, 0],
        [0, 0, 3, 4, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]))

    assert np.all(ds[1] == np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 1, 2, 0, 0, 0],
        [0, 3, 4, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]))


def test_transforms(tmpdir):
    p = _create_simple_test_img(tmpdir)

    ds = CropDataset(p, crop_size=2, transform=transforms.ToTensor())
    assert torch.allclose(ds[0], torch.tensor([[[1, 2]], [[3, 4]]], dtype=torch.float).T / 255)
