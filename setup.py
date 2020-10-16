
from setuptools import setup, find_packages
from os import path as osp

import sys
from io import open

here = osp.abspath(osp.dirname(__file__))
parent = osp.abspath(osp.dirname(here))
sys.path.insert(0, parent)
sys.path.insert(1, here)
from version import __version__

# Get the long description from the README file
with open(osp.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='estimateMI',
      version=__version__,
      description='Estimating differential entropy using Gaussian convolution',
      url='https://github.com/ankithmo/estimateMI',
      author='Ankith Mohan',
      author_email='ankithmo@usc.edu',
      keywords=['pytorch', 'deep neural networks', 'differential entropy estimation', 'mutual information estimation', 'information flow in deep neural networks'],
      long_description=long_description,
      long_description_content_type='text/markdown',
      license='GPL-3',
      include_package_data=True,
      classifiers=[
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: GPL-3.0 License',
    ],
)