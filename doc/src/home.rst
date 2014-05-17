****
Home
****

Overview
========

What is PyFR?
-------------

PyFR is an open-source Python based framework for solving advection-diffusion
type problems on streaming architectures using the Flux Reconstruction approach
of Huynh. The framework is designed to solve a range of governing systems on
mixed unstructured grids containing various element types. It is also designed
to target a range of hardware platforms via use of an in-built domain specific
language derived from the Mako templating engine. The current release (PyFR
|release|) has the following capabilities:

- Governing equations - Euler, Navier Stokes
- Dimensionality - 2D, 3D
- Element types - Triangles, Quadrilaterals, Hexahedra, Prisms, Tetrahedra
- Platforms - CPU clusters, Nvidia GPU clusters, AMD GPU clusters
- Spatial discretisation - High-order flux reconstruction
- Temporal discretisation - Explicit Runge-Kutta schemes
- Precision - Single, Double
- Mesh files read - Gmsh (.msh)
- Solution files produced - Unstructured VTK (.vtu)

What is PyFR Not?
-----------------

PyFR is not a fully fledged 'production' flow solver with all the associated
bells and whistles. Additionally, while we will do our best to help new users,
no level of support is guaranteed!

Who is Developing PyFR?
-----------------------

PyFR is being developed in the `Vincent Lab
<https://www.imperial.ac.uk/aeronautics/research/vincentlab/>`_, Department of
Aeronautics, Imperial College London, UK. More details about the development
team are available `here <http://www.pyfr.org/team.php>`__.

How do I get PyFR?
------------------

PyFR is available for free under an open-source license. You can download it
`here <http://www.pyfr.org/download.php>`__.

How do I Cite PyFR?
-------------------

If you would like to cite PyFR, please reference the following paper:

- `PyFR: An Open Source Framework for Solving Advection-Diffusion Type Problems on Streaming Architectures using the Flux Reconstruction Approach. F. D. Witherden, A. M. Farrington, P. E. Vincent <http://arxiv.org/abs/1312.1638>`_.

Who is Funding PyFR?
--------------------

Development of PyFR is supported by the `Engineering and Physical Sciences
Research Council <http://www.epsrc.ac.uk/>`_ and `Airbus
<http://www.airbus.com/>`_. We are also grateful for hardware donations from
Nvidia, Intel, and AMD.
