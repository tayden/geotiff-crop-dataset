from setuptools import setup

setup(
    name='geotiff-crop-dataset',
    url='https://github.com/tayden/pytorch_crop_dataset',
    author='Taylor Denouden',
    author_email='taylor.denouden@hakai.org',
    description='A Pytorch dataset that crops rasterio-readable image files to small sections on the fly.',
    install_requires=[
        'numpy',
        'rasterio',
        'setuptools-scm'
    ],
    license='MIT',
    packages=['geotiff_crop_dataset'],
    setup_requires=['setuptools_scm'],
    use_scm_version=True,
)
