import numpy as np
import rasterio
import torch
from torchvision import transforms

from geotiff_crop_dataset import CropDatasetReader


def _create_simple_1band_img(tmpdir):
    simple_img = np.array([
        [1, 2],
        [3, 4]
    ])

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


def _create_simple_3band_img(tmpdir):
    simple_img = np.array([[
        [11, 112, 213],
        [221, 22, 123]
    ], [
        [131, 232, 33],
        [41, 142, 243]
    ]])

    p = str(tmpdir.mkdir("simple").join("simple_img.tif"))
    with rasterio.open(
            p,
            'w',
            driver='GTiff',
            height=simple_img.shape[0],
            width=simple_img.shape[1],
            count=3,
            dtype=rasterio.uint8) as dst:
        for b in range(simple_img.shape[-1]):
            dst.write(simple_img[:, :, b].astype(rasterio.uint8), b + 1)

    return p


def test_simple_crop(tmpdir):
    p = _create_simple_1band_img(tmpdir)

    with CropDatasetReader(p, crop_size=1) as ds:
        assert ds[0] == np.array([[1]])
        assert ds[1] == np.array([[2]])
        assert ds[2] == np.array([[3]])
        assert ds[3] == np.array([[4]])

    with CropDatasetReader(p, crop_size=2) as ds:
        assert np.all(ds[0] == np.array([[1, 2], [3, 4]]))


def test_padding(tmpdir):
    p = _create_simple_1band_img(tmpdir)
    with CropDatasetReader(p, crop_size=1, padding=1, fill_value=0) as ds:
        assert np.all(ds[0] == np.array([[0, 0, 0], [0, 1, 2], [0, 3, 4]]))
        assert np.all(ds[1] == np.array([[0, 0, 0], [1, 2, 0], [3, 4, 0]]))
        assert np.all(ds[2] == np.array([[0, 1, 2], [0, 3, 4], [0, 0, 0]]))
        assert np.all(ds[3] == np.array([[1, 2, 0], [3, 4, 0], [0, 0, 0]]))

    with CropDatasetReader(p, crop_size=2, padding=1, fill_value=0) as ds:
        assert np.all(ds[0] == np.array([[0, 0, 0, 0],
                                         [0, 1, 2, 0],
                                         [0, 3, 4, 0],
                                         [0, 0, 0, 0]
                                         ]))

    with CropDatasetReader(p, crop_size=2, padding=2, fill_value=8) as ds:
        assert np.all(ds[0] == np.array([
            [8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8],
            [8, 8, 1, 2, 8, 8],
            [8, 8, 3, 4, 8, 8],
            [8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8],
        ]))


def test_stride(tmpdir):
    p = _create_simple_1band_img(tmpdir)
    with CropDatasetReader(p, crop_size=2, stride=1, padding=2, fill_value=0) as ds:
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


def test_fill_values(tmpdir):
    p = _create_simple_1band_img(tmpdir)
    with CropDatasetReader(p, crop_size=2, stride=1, padding=2) as ds:
        # Should default to 0
        assert ds.raster.nodata is None
        assert np.all(ds[0] == np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 2, 0, 0],
            [0, 0, 3, 4, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]))

    # Should use raster nodata if available
    simple_img = np.array([
        [1, 2],
        [3, 4]
    ])

    p = str(tmpdir.mkdir("nodata").join("simple_img.tif"))
    with rasterio.open(
            p,
            'w',
            driver='GTiff',
            height=simple_img.shape[0],
            width=simple_img.shape[1],
            count=1,
            nodata=9,
            dtype=rasterio.uint8) as dst:
        dst.write(simple_img.astype(rasterio.uint8), 1)

    with CropDatasetReader(p, crop_size=2, stride=1, padding=2) as ds:
        assert ds.raster.nodata == 9
        assert np.all(ds[0] == np.array([
            [9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9],
            [9, 9, 1, 2, 9, 9],
            [9, 9, 3, 4, 9, 9],
            [9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9],
        ]))

    with CropDatasetReader(p, crop_size=2, stride=1, padding=2, fill_value=7) as ds:
        # Should use value defined at class instantiation
        assert ds.raster.nodata == 9
        assert np.all(ds[0] == np.array([
            [7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7],
            [7, 7, 1, 2, 7, 7],
            [7, 7, 3, 4, 7, 7],
            [7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7],
        ]))


def test_transforms(tmpdir):
    p = _create_simple_1band_img(tmpdir)
    with CropDatasetReader(p, crop_size=2, transform=transforms.ToTensor()) as ds:
        it = (ds[0] * 255).to(torch.long)
        gt = torch.tensor([[1, 2], [3, 4]])

        assert torch.allclose(it, gt)


def test_band_ordering(tmpdir):
    p = _create_simple_3band_img(tmpdir)
    with CropDatasetReader(p, crop_size=2, transform=transforms.ToTensor()) as ds:
        it = (ds[0] * 255).to(torch.long)
        gt = torch.tensor([[
            [11, 112, 213],
            [221, 22, 123]
        ], [
            [131, 232, 33],
            [41, 142, 243]
        ]]).permute(2, 0, 1)  # Rearrange to (C, H, W)

        assert it.shape == torch.Size([3, 2, 2])
        assert torch.allclose(it, gt)


def test_len(tmpdir):
    p = _create_simple_3band_img(tmpdir)
    with CropDatasetReader(p, crop_size=1) as ds:
        assert len(ds) == 4

    with CropDatasetReader(p, crop_size=2) as ds:
        assert len(ds) == 1


def test_y0x0_getters(tmpdir):
    p = _create_simple_3band_img(tmpdir)
    with CropDatasetReader(p, crop_size=1) as ds:
        assert ds.y0 == [0, 0, 1, 1]
        assert ds.x0 == [0, 1, 0, 1]
