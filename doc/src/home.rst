****
Home
****

Overview
========

What is PyFR?
-------------
PyFR is an open-source Python based framework for solving
advection-diffusion type problems on streaming architectures using the
Flux Reconstruction approach of Huynh. The framework is designed to
solve a range of governing systems on mixed unstructured grids
containing various element types. It is also designed to target a range
of hardware platforms via use of an in-built domain specific language
derived from the Mako templating engine. The current release (PyFR
|release|) has the following capabilities:

- Governing Equations - Euler, Navier Stokes
- Dimensionality - 2D, 3D
- Element Types - Triangles, Quadrilaterals, Hexahedra, Prisms,
  Tetrahedra, Pyramids
- Platforms - CPU Clusters, Nvidia GPU Clusters, AMD GPU Clusters, Intel
  Xeon Phi Clusters
- Spatial Discretisation - High-Order Flux Reconstruction
- Temporal Discretisation - Explicit Runge-Kutta
- Precision - Single, Double
- Mesh Files Imported - Gmsh (.msh), CGNS (.cgns)
- Solution Files Exported - Unstructured VTK (.vtu, .pvtu)

Who is Developing PyFR?
-----------------------

PyFR is being developed in the `Vincent Lab
<https://www.imperial.ac.uk/aeronautics/research/vincentlab/>`_, 
Department of Aeronautics, Imperial College London, UK. More details 
about the development team are available 
`here <http://www.pyfr.org/team.php>`_.

How do I Cite PyFR?
-------------------

To cite PyFR, please reference the following paper:

- `PyFR: An Open Source Framework for Solving Advection-Diffusion Type 
  Problems on Streaming Architectures using the Flux Reconstruction 
  Approach. F. D. Witherden, A. M. Farrington, P. E. Vincent. Computer 
  Physics Communications, Volume 185, Pages 3028-3040, 2014. 
  <http://www.sciencedirect.com/science/article/pii/S0010465514002549>`_

Who is Funding PyFR?
--------------------

Development of PyFR is supported by the `Engineering and Physical 
Sciences Research Council <http://www.epsrc.ac.uk/>`_, `Innovate UK
<https://www.gov.uk/government/organisations/innovate-uk>`_, the
`European Commission
<http://ec.europa.eu/programmes/horizon2020/>`_,
`BAE Systems <http://www.baesystems.com/>`_, and
`Airbus <http://www.airbus.com/>`_. We are also grateful for hardware
donations from Nvidia, Intel, and AMD.
