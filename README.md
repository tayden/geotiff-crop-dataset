# GeoTiff Crop Dataset

Create PyTorch Datasets from GeoTiff files

## Installation
`pip install geotiff-crop-dataset`

## Usage
```python
from torchvision import transforms

from geotiff_crop_dataset import CropDataset

ds = CropDataset(
    "./path/to/geotiff.tif", 
    crop_size=32,  # Edge size of each cropped square section
    stride=16,  # Number of pixels between each cropped sub-image
    padding=2,  # Number of pixels appended to sides of cropped images
    fill_value=0,  # The value to use for nodata sections and padded regions
    transform=transforms.ToTensor()  # torchvision transform functions
)
```

Then use the dataset like any other Pytorch dataset

```python
import torch
from torch.utils.data import DataLoader

from geotiff_crop_dataset import CropDataset

ds = CropDataset(...)
batch_size = 8
dataloader = DataLoader(ds, batch_size=batch_size, num_workers=4, pin_memory=True)

# Use the cropped sections during training or inference
for i, x in enumerate(dataloader):
    x = x.to(torch.device('cuda'))

    # Get cropped section origin in the original image
    y0x0s = ds.y0x0[i*batch_size: i*batch_size+batch_size]
    
    # Or do
    y0s = ds.y0[i*batch_size: i*batch_size+batch_size]
    x0s = ds.x0[i*batch_size: i*batch_size+batch_size]

    ...
```