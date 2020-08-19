from setuptools import setup

setup(
    name='geotiff-crop-dataset',
    version='0.0.1rc6',
    packages=['geotiff_crop_dataset'],
    url='https://github.com/tayden/pytorch_crop_dataset',
    install_requires=[
        'rasterio',
        'numpy'
    ],
    license='MIT',
    author='Taylor Denouden',
    author_email='taylor.denouden@hakai.org',
    description='A Pytorch dataset that crops rasterio-readable image files to small sections on the fly.'
)
