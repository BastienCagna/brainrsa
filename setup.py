#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import re


def get_property(prop, project):
    result = re.findall(r'^{}\s*=\s*[\'"]([^\'"]*)[\'"]$'.format(
        prop), open(project + '/__init__.py').read(), re.MULTILINE)
    assert type(result[0]) == str
    return result[0]


setup(
    name='brainrsa',
    version=get_property('__version__', 'brainrsa'),
    packages=['brainrsa'],
    url='https://github.com/BastienCagna/brainrsa',
    license='',
    author='Bastien Cagna',
    author_email='bastiencagna@gmail.com',
    description='A Python package to do Brain Representational Similarity Analysis',
    install_requires=["numpy", "matplotlib", "seaborn",
                      "sklearn", "scipy", "pandas", "nilearn==0.6"]
)
