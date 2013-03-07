========
Overview
========

Introduction
------------

PyFR is a Python based high-order compressible fluid flow solver based
on energy stable Vincent-Castonguay-Jameson-Huynh schemes [VCJ2011]_. It
is currently being developed in the department of Aeronautics at
`Imperial College London <https://www.imperial.ac.uk>`_ under the
direction of Dr. Peter Vincent.

Capabilities
------------

PyFR has the folllowing capabilities:

- Governing equations - Euler, Navier Stokes
- Dimensionality - 3D
- Element types - Hexahedra
- Platforms - Nvidia GPUs
- Spatial discretisation - Arbitrary order Vincent-Castonguay-Jameson-Huynh schemes
- Temporal discretisation - Explicit Runge-Kutta schemes
- Mesh files read - Gmsh (.msh)
- Solution files written - Paraview (.vtu)

Coding Ethos
--------------

Authors
-------

See the AUTHORS file for a complete list of authors.

Licensing
---------

New BSD License (see the LICENSE file for details).

Funding
-------

Development of PyFR is supported by the Engineering and Physical
Sciences Research Council.

.. [VCJ2011] Vincent, P. E., Castonguay, P., & Jameson, A. (2011). A new
   class of high-order energy stable flux reconstruction schemes. J Sci
   Comput, 47(1), 50-72.
