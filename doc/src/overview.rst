********
Overview
********

Introduction
============

PyFR is a Python based high-order compressible fluid flow solver based
on energy stable Vincent-Castonguay-Jameson-Huynh (VCJH) schemes
[VCJ2011]_. It is currently being developed in the department of
Aeronautics at `Imperial College London <https://www.imperial.ac.uk>`_
under the direction of Dr. Peter Vincent.

Ethos
=====

Our objective is to develop a compact and efficient codebase that can
target multiple hardware platforms. High-level platform independent code
is written in Python. The Python code in-turn targets multiple hardware
platfroms via a range of platfrom specific 'backend' kernels. Due to how
VCJH schemes [VCJ2011]_ have been abstracted, the majority of backend
kernels consitute only simple matrix multiply operations.

Capabilities
============

PyFR has the folllowing capabilities:

- Governing equations - Euler, Navier Stokes
- Dimensionality - 3D
- Element types - Hexahedra
- Platforms - Nvidia GPUs
- Spatial discretisation - Arbitrary order Vincent-Castonguay-Jameson-Huynh schemes
- Temporal discretisation - Explicit Runge-Kutta schemes
- Mesh files read - Gmsh (.msh)
- Solution files written - Paraview (.vtu)

Authors
=======

See the AUTHORS file for full details.

Licensing
=========

See the LICENSE file for full details.

Funding
=======

Development of PyFR is supported by the Engineering and Physical
Sciences Research Council.

.. [VCJ2011] Vincent, P. E., Castonguay, P., & Jameson, A. (2011). A new
   class of high-order energy stable flux reconstruction schemes. J Sci
   Comput, 47(1), 50-72.
