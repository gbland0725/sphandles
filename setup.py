#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', ]

setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="Garret Bland",
    author_email='garretbland@gmail.com',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Sphandles contains a machine-learning algorithm to differentiate Ti-containing natural and engineered NMs measured by spICP-TOFMS data. It also contains functionality to parse spICP-TOFMS dataframes and generate easy-to-use figures for single particle analysis.",
    entry_points={
        'console_scripts': [
            'sphandles=sphandles.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='sphandles',
    name='sphandles',
    packages=find_packages(include=['sphandles', 'sphandles.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/gbland0725/sphandles',
    version='0.1.0',
    zip_safe=False,
)
