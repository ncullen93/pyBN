#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages

setup(name='pyBN',
      version='0.1',
      description='Bayesian Networks in Python',
      author='Nicholas Cullen',
      author_email='ncullen.th@dartmouth.edu',
      url='sites.dartmouth.edu/ncullen',
      packages=find_packages()
     )