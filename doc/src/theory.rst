******
Theory
******

Overview
========

High-order numerical methods for unstructured grids combine the superior
accuracy of high-order spectral or finite difference methods with the geometric
flexibility of low-order finite volume or finite element schemes. The Flux
Reconstruction (FR) approach, first proposed by Huynh [H2007]_, unifies various
high-order schemes for unstructured grids within a single framework.
Additionally, the FR approach exhibits a significant degree of element locality,
and is thus able to run efficiently on modern streaming architectures, such as
Graphical Processing Units (GPUs). The aforementioned properties of FR mean it
offers a promising route to performing affordable, and hence industrially
relevant, scale-resolving simulations of hitherto intractable unsteady flows
(involving separation, acoustics etc.) within the vicinity of real-world
engineering geometries [VJ2011]_.

PyFR employs energy-stable Vincent-Castonguay-Jameson-Huynh (VCJH) type FR
schemes [VCJ2011]_

.. [H2007] Huynh, H. T. (2007). A flux reconstruction approach to
   high-order schemes including discontinuous Galerkin methods. AIAA
   Paper 2007-4079.

.. [VJ2011] Vincent, P. E., & Jameson, A. (2011). Facilitating the
   adoption of unstructured high-order methods amongst a wider community
   of fluid dynamicists. Math Mod Nat Phenom, 6(3), 97-140.

.. [VCJ2011] Vincent, P. E., Castonguay, P., & Jameson, A. (2011). A new
   class of high-order energy stable flux reconstruction schemes. J Sci
   Comput, 47(1), 50-72.
