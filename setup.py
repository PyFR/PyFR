#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

import sys

# Python version
if sys.version_info[:2] < (2,7):
    print('PyFR requires Python 2.7 or newer.  Python {}.{} detected' %\
          sys.version_info[:2])
    sys.exit(-1)

# Modules
modules = [
    'pyfr.backends',
    'pyfr.backends.cuda',
    'pyfr.elements']

# Tests
tests = [
    'pyfr.elements.tests']

# Data
package_data = {
    'pyfr.backends.cuda': ['kernels/*'],
    'pyfr.elements.tests': ['*.npz']}

# Dependencies
install_requires = [
    'pycuda >= 2011.2',
    'mpi4py >= 1.3',
    'mako',
    'numpy',
    'sympy']

# Info
classifiers = [
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2.7',
    'Topic :: Scientific/Engineering']

long_description = '''PyFR is a...'''

setup(
    name='pyfr',
    version='0.1',
    description='Flux reconstruction in Python',
    long_description=long_description,
    author='Imperial College London',
    license='BSD',
    keywords='Math',
    packages=['pyfr'] + modules + tests,
    package_data=package_data,
    install_requires=install_requires,
    classifiers=classifiers
)
