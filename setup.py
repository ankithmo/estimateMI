
from setuptools import setup, find_packages
from os import path as osp

import sys
from io import open

here = osp.abspath(osp.dirname(__file__))
sys.path.insert(0, osp.join(here, 'dRBM'))
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
      install_requires = [
        'torch>=1.5.0',
        'numpy>=1.13.3',
        'future>=0.16.0',
        'pillow>=4.1.1',
        'scipy>=1.4.1',
        'tqdm>=4.41.1',
        'scikit-learn>=0.22.2',
        'numba>=0.48.0',
        'requests>=2.23.0',
        'pandas>=1.0.5',
        'h5py>=2.10.0',
        'jinja2>=2.11.2',
        'decorator>=4.3.0',
        'joblib>=0.11.0',
        'llvmlite<0.32.0',
        'setuptools>=49.6.0',
        'chardet<4,>=3.0.2',
        'urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1',
        'idna<3,>=2.5',
        'certifi>=2017.4.17',
        'python-dateutil>=2.6.1',
        'pytz>=2017.2',
        'six>=1.15.0',
        'pyparsing>=2.4.7',
        'matplotlib>=2.0.0',
        'MarkupSafe>=0.23',
        'cycler>=0.10',
        'kiwisolver>=1.0.1'
      ],
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