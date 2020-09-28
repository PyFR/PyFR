# PyFR

## Overview

PyFR is an open-source Python based framework for solving advection-diffusion
type problems on streaming architectures using the Flux Reconstruction
approach of Huynh. The framework is designed to solve a range of governing
systems on mixed unstructured grids containing various element types. It is
also designed to target a range of hardware platforms via use of an in-built
domain specific language derived from the Mako templating engine.

## Examples

Examples of using PyFR are available in the `examples` directory. Currently
available examples are:

- 2D Couette flow
- 2D Euler vortex
- 2D incompressible cylinder flow

## Contributing

To contribute to PyFR please follow the steps listed below:

1. Fork this repository to your GitHub account
2. Create a new branch in your forked repository
3. Make changes in your new branch
4. Submit your changes by creating a Pull Request to the `develop` branch of the original PyFR repository

Modifications to the `develop` branch are eventually merged into the master
branch for a new release.

## Authors

See the AUTHORS file.

## License

PyFR is released under the New BSD License (see the LICENSE file for details).
Documentation is made available under a Creative Commons Attribution 4.0
license (see <http://creativecommons.org/licenses/by/4.0/>).
