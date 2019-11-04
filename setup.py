#!/usr/bin/env python3
from setuptools import setup, find_packages
import sys

if sys.version_info < (3,):
    sys.exit('Sorry, Python3 is required for dl_ext.')

setup(
    name='dl_ext',
    version='1.0.1',
    description='Chen Linghao\'s personal utils.',
    packages=find_packages(),
    install_requires=[
        #         'torch',
        'matplotlib',
        'numpy',
        'visdom',
        'progressbar2',
        'torchvision',
        'fastai',
        'multipledispatch',
        'imageio',
        # 'maskrcnn_benchmark',
        'requests'
    ],
)
# to install this, use 'python setup.py build develop'
# to add a new module, write a script in myutils/myutils directory, then import it in __init__.py, then install it .
# 	• Build: python setup.py bdist_wheel
# 	• Upload: python -m twine upload dist/*
# Build and upload
#   python setup.py bdist_wheel && python -m twine upload dist/*
# Usage:
# Documentation or tutorial has not been fully built, please see comments in source code for help.
