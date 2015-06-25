#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from setuptools import setup
import sys


# Python version
if sys.version_info[:2] < (3, 3):
    print('PyFR requires Python 3.3 or newer')
    sys.exit(-1)

# PyFR version
vfile = open('pyfr/_version.py').read()
vsrch = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", vfile, re.M)

if vsrch:
    version = vsrch.group(1)
else:
    print('Unable to find a version string in pyfr/_version.py')

# Modules
modules = [
    'pyfr.backends',
    'pyfr.backends.base',
    'pyfr.backends.cuda',
    'pyfr.backends.cuda.kernels',
    'pyfr.backends.opencl',
    'pyfr.backends.opencl.kernels',
    'pyfr.backends.openmp',
    'pyfr.backends.openmp.kernels',
    'pyfr.integrators',
    'pyfr.plugins',
    'pyfr.quadrules',
    'pyfr.readers',
    'pyfr.partitioners',
    'pyfr.scripts',
    'pyfr.solvers',
    'pyfr.solvers.base',
    'pyfr.solvers.baseadvec',
    'pyfr.solvers.baseadvec.kernels',
    'pyfr.solvers.baseadvecdiff',
    'pyfr.solvers.baseadvecdiff.kernels',
    'pyfr.solvers.euler',
    'pyfr.solvers.euler.kernels',
    'pyfr.solvers.euler.kernels.bcs',
    'pyfr.solvers.euler.kernels.rsolvers',
    'pyfr.solvers.navstokes',
    'pyfr.solvers.navstokes.kernels',
    'pyfr.solvers.navstokes.kernels.bcs',
    'pyfr.writers'
]

# Tests
tests = [
    'pyfr.tests'
]

# Data
package_data = {
    'pyfr.backends.cuda.kernels': ['*.mako'],
    'pyfr.backends.opencl.kernels': ['*.mako'],
    'pyfr.backends.openmp.kernels': ['*.mako'],
    'pyfr.quadrules': [
        'hex/*.txt',
        'line/*.txt',
        'pri/*.txt',
        'pyr/*.txt',
        'quad/*.txt',
        'tet/*.txt',
        'tri/*.txt'
    ],
    'pyfr.solvers.baseadvec.kernels': ['*.mako'],
    'pyfr.solvers.baseadvecdiff.kernels': ['*.mako'],
    'pyfr.solvers.euler.kernels': ['*.mako'],
    'pyfr.solvers.euler.kernels.bcs': ['*.mako'],
    'pyfr.solvers.euler.kernels.rsolvers': ['*.mako'],
    'pyfr.solvers.navstokes.kernels': ['*.mako'],
    'pyfr.solvers.navstokes.kernels.bcs': ['*.mako'],
    'pyfr.tests': ['*.npz']
}

# Additional data
data_files = [
    ('', ['pyfr/__main__.py'])
]

# Hard dependencies
install_requires = [
    'h5py >= 2.4',
    'mako >= 1.0.0',
    'mpi4py >= 1.3',
    'mpmath >= 0.18',
    'numpy >= 1.8',
    'pytools >= 2014.3'
]

# Soft dependencies
extras_require = {
    'cuda': ['pycuda >= 2011.2'],
    'opencl': ['pyopencl >= 2013.2']
}

# Scripts
console_scripts = [
    'pyfr = pyfr.scripts.main:main'
]

# Info
classifiers = [
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3.3',
    'Topic :: Scientific/Engineering'
]

long_description = '''PyFR is an open-source Python based framework for
solving advection-diffusion type problems on streaming architectures
using the Flux Reconstruction approach of Huynh. The framework is
designed to solve a range of governing systems on mixed unstructured
grids containing various element types. It is also designed to target a
range of hardware platforms via use of an in-built domain specific
language derived from the Mako templating engine. PyFR is being
developed in the Vincent Lab, Department of Aeronautics, Imperial
College London, UK.'''

setup(name='pyfr',
      version=version,
      description='Flux Reconstruction in Python',
      long_description=long_description,
      author='Imperial College London',
      author_email='info@pyfr.org',
      url='http://www.pyfr.org/',
      license='BSD',
      keywords='Math',
      packages=['pyfr'] + modules + tests,
      package_data=package_data,
      data_files=data_files,
      entry_points={'console_scripts': console_scripts},
      install_requires=install_requires,
      extras_require=extras_require,
      classifiers=classifiers
)
