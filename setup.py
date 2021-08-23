#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import io
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    with io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ) as fh:
        return fh.read()


setup(
    name='sphandles',
    version='0.1.24',
    license='MIT License',
    description='Sphandles contains a machine-learning algorithm to differentiate Ti-containing natural and engineered NMs measured by spICP-TOFMS data. It also contains functionality to parse spICP-TOFMS dataframes and generate easy-to-use figures for single particle analysis.',
    author='Garret Bland',
    author_email='garretbland@gmail.com',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
	'Intended Audience :: Science/Research',
	'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
	'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Utilities',
    ],
    keywords=[
        'spICP-TOFMS', 'nanoparticles', 'classification',
    ],
    install_requires= ['numpy',
	'pandas',
	'pytest',
	'scikit-learn',
	'statsmodels',
	'matplotlib',
	'seaborn',
	'scipy',
	'click'
                      ],
    setup_requires=[
        # 'pytest-runner',
    ],
    entry_points={
        'console_scripts': [
            'sphandles = sphandles.click:main',
        ]
    },
)