#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DeepSeek-OCR: Contexts Optical Compression
A model to investigate the role of vision encoders from an LLM-centric viewpoint.
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='deepseek-ocr',
    version='1.0.0',
    author='DeepSeek AI',
    author_email='support@deepseek.com',
    description='DeepSeek-OCR: Contexts Optical Compression',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/deepseek-ai/DeepSeek-OCR',
    project_urls={
        'Bug Tracker': 'https://github.com/deepseek-ai/DeepSeek-OCR/issues',
        'Documentation': 'https://github.com/deepseek-ai/DeepSeek-OCR',
        'Source Code': 'https://github.com/deepseek-ai/DeepSeek-OCR',
        'Paper': 'https://arxiv.org/abs/2510.18234',
    },
    packages=find_packages(where='DeepSeek-OCR-master'),
    package_dir={'': 'DeepSeek-OCR-master'},
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'mypy>=0.950',
        ],
        'vllm': [
            'vllm>=0.8.5',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Text Processing :: Linguistic',
    ],
    keywords='deepseek ocr vision language model multimodal ai machine-learning',
    include_package_data=True,
    zip_safe=False,
)
