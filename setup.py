#!/usr/bin/env python
from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='antenna_pattern',
    version='0.1.0',
    description='Core functionality for antenna pattern analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Justin Long',
    author_email='justinwlong1@gmail.com',
    url='https://github.com/freespacemind/antenna_pattern',
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'xarray>=0.19.0',
        'matplotlib>=3.4.0',  # For visualizations
        'PyQt6>=6.4.0',       # For GUI
    ],
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'pytest-cov>=2.10.0',
            'black>=20.8b1',
            'flake8>=3.8.0',
            'isort>=5.0.0',
        ],
    },
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    keywords='antenna, electromagnetics, pattern, radiation, analysis',
    entry_points={
        'console_scripts': [
            'antenna-pattern-gui=antenna_pattern.gui.run_gui:main',
        ],
    },
)