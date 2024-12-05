#!/usr/bin/env python
import re
from setuptools import setup
import sys


# Check Python version
if sys.version_info < (3, 11):
    sys.exit('Minimum Python version is 3.11')


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
    'pyfr.backends.hip',
    'pyfr.backends.hip.kernels',
    'pyfr.backends.metal',
    'pyfr.backends.metal.kernels',
    'pyfr.backends.opencl',
    'pyfr.backends.opencl.kernels',
    'pyfr.backends.openmp',
    'pyfr.backends.openmp.kernels',
    'pyfr.integrators',
    'pyfr.integrators.dual',
    'pyfr.integrators.dual.phys',
    'pyfr.integrators.dual.pseudo',
    'pyfr.integrators.dual.pseudo.kernels',
    'pyfr.integrators.std',
    'pyfr.integrators.std.kernels',
    'pyfr.partitioners',
    'pyfr.plugins',
    'pyfr.plugins.kernels',
    'pyfr.quadrules',
    'pyfr.readers',
    'pyfr.solvers',
    'pyfr.solvers.aceuler',
    'pyfr.solvers.aceuler.kernels',
    'pyfr.solvers.aceuler.kernels.bcs',
    'pyfr.solvers.aceuler.kernels.rsolvers',
    'pyfr.solvers.acnavstokes',
    'pyfr.solvers.acnavstokes.kernels',
    'pyfr.solvers.acnavstokes.kernels.bcs',
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
    'pyfr.writers',
    'pyfr.writers.vtk'
]

# Tests
tests = [
    'pyfr.tests'
]

# Data
package_data = {
    'pyfr.backends.cuda.kernels': ['*.mako'],
    'pyfr.backends.hip.kernels': ['*.mako'],
    'pyfr.backends.metal.kernels': ['*.mako'],
    'pyfr.backends.opencl.kernels': ['*.mako'],
    'pyfr.backends.openmp.kernels': ['*.mako'],
    'pyfr.integrators.dual.pseudo.kernels': ['*.mako'],
    'pyfr.integrators.std.kernels': ['*.mako'],
    'pyfr.plugins.kernels': ['*.mako'],
    'pyfr.quadrules': [
        'hex/*.txt',
        'line/*.txt',
        'pri/*.txt',
        'pyr/*.txt',
        'quad/*.txt',
        'tet/*.txt',
        'tri/*.txt'
    ],
    'pyfr.solvers.aceuler.kernels': ['*.mako'],
    'pyfr.solvers.aceuler.kernels.bcs': ['*.mako'],
    'pyfr.solvers.aceuler.kernels.rsolvers': ['*.mako'],
    'pyfr.solvers.acnavstokes.kernels': ['*.mako'],
    'pyfr.solvers.acnavstokes.kernels.bcs': ['*.mako'],
    'pyfr.solvers.baseadvec.kernels': ['*.mako'],
    'pyfr.solvers.baseadvecdiff.kernels': ['*.mako'],
    'pyfr.solvers.euler.kernels': ['*.mako'],
    'pyfr.solvers.euler.kernels.bcs': ['*.mako'],
    'pyfr.solvers.euler.kernels.rsolvers': ['*.mako'],
    'pyfr.solvers.navstokes.kernels': ['*.mako'],
    'pyfr.solvers.navstokes.kernels.bcs': ['*.mako'],
    'pyfr.tests': ['*.npz']
}

# Hard dependencies
install_requires = [
    'gimmik >= 3.2.1',
    'h5py >= 2.10',
    'mako >= 1.0.0',
    'mpi4py >= 4.0.0',
    'numpy >= 1.26.4',
    'platformdirs >= 2.2.0',
    'pytools >= 2016.2.1',
    'rtree >= 1.0.1'
]

# Soft dependencies
extras_require = {
    'metal': ['pyobjc-framework-Metal >= 9.0']
}

# Scripts
console_scripts = [
    'pyfr = pyfr.__main__:main'
]

# Info
classifiers = [
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12',
    'Programming Language :: Python :: 3.13',
    'Topic :: Scientific/Engineering'
]

long_description = '''PyFR is an open-source Python based framework for
solving advection-diffusion type problems on streaming architectures
using the Flux Reconstruction approach of Huynh. The framework is
designed to solve a range of governing systems on mixed unstructured
grids containing various element types. It is also designed to target a
range of hardware platforms via use of an in-built domain specific
language derived from the Mako templating engine.'''

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
      entry_points={'console_scripts': console_scripts},
      python_requires='>=3.11',
      install_requires=install_requires,
      extras_require=extras_require,
      classifiers=classifiers)
