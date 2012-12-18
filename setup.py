#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

import sys
import re

# Python version
if sys.version_info[:2] < (2,7):
    print('PyFR requires Python 2.7 or newer.  Python {}.{} detected'\
          .format(*sys.version_info[:2]))
    sys.exit(-1)

# PyFR version
vfile = open('pyfr/_version.py').read()
vsrch = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", vstr, re.M)

if vsrch:
    version = vsrch.group(1)
else:
    print 'Unable to find a version string in pyfr/_version.py'

# Modules
modules = [
    'pyfr.backends',
    'pyfr.backends.cuda',
    'pyfr.bases',
    'pyfr.integrators'
    'pyfr.scripts']

# Tests
tests = [
    'pyfr.tests']

# Data
package_data = {
    'pyfr.backends.cuda': ['kernels/*'],
    'pyfr.tests': ['*.npz']}

# Dependencies
install_requires = [
    'pycuda >= 2011.2',
    'mpi4py >= 1.3',
    'mako',
    'numpy >= 1.6',
    'sympy >= 0.7.2']

# Scripts
console_scripts = [
    'pyfr-mesh = pyfr.scripts.mesh:main',
    'pyfr-sim = pyfr.scripts.sim:main',
    'pyfr-postp = pyfr.scripts.postp:main']

# Info
classifiers = [
    'License :: OSI Approved :: New BSD License',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2.7',
    'Topic :: Scientific/Engineering']

long_description = '''PyFR is a Python based high-order compressible
fluid flow solver based on energy stable
Vincent-Castonguay-Jameson-Huynh schemes. It is currently being
developed in the department of Aeronautics at Imperial College London
under the direction of Dr. Peter Vincent.'''

setup(
    name='pyfr',
    version=version,
    description='Flux Reconstruction in Python',
    long_description=long_description,
    author='Imperial College London',
    license='BSD',
    keywords='Math',
    packages=['pyfr'] + modules + tests,
    package_data=package_data,
    entry_points={'console_scripts': console_scripts},
    install_requires=install_requires,
    classifiers=classifiers
)
