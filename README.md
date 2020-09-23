# PyFR

## Overview

PyFR is an open-source Python based framework for solving advection-diffusion type problems on streaming architectures using the Flux Reconstruction approach of Huynh. The framework is designed to solve a range of governing systems on mixed unstructured grids containing various element types. It is also designed to target a range of hardware platforms via use of an in-built domain specific language derived from the Mako templating engine.

## Examples

Examples of using PyFR are available in the `examples` directory. A
`README.md` is provided for each example with instructions on how to run the
example. Currently available examples are:

- 2D Couette flow
- 2D Euler vortex
- 2D incompressible cylinder flow

## Contributing

The first step to contribute to PyFR is to fork this repository. In your
personal fork, you can make changes in the `develop` branch or create another
branch. Submit a Pull Request from your forked repository when your work is
ready to be added to the official PyFR repository. The Pull Request should be
made to the `develop` branch of the original PyFR repository.

## Authors

See the AUTHORS file.

## License

PyFR is released under the New BSD License (see the LICENSE file for details).
Documentation is made available under a Creative Commons Attribution 4.0
license (see <http://creativecommons.org/licenses/by/4.0/>).
