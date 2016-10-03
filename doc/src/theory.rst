******
Theory
******

Flux Reconstruction
===================

Overview
--------

High-order numerical methods for unstructured grids combine the
superior accuracy of high-order spectral or finite difference methods
with the geometrical flexibility of low-order finite volume or finite
element schemes. The Flux Reconstruction (FR) approach unifies various
high-order schemes for unstructured grids within a single framework.
Additionally, the FR approach exhibits a significant degree of element
locality, and is thus able to run efficiently on modern streaming
architectures, such as Graphical Processing Units (GPUs). The
aforementioned properties of FR mean it offers a promising route to
performing affordable, and hence industrially relevant, scale-resolving
simulations of hitherto intractable unsteady flows (involving
separation, acoustics etc.) within the vicinity of real-world
engineering geometries. An detailed overview of the FR approach is
given in:

- `A Flux Reconstruction Approach to High-Order Schemes Including 
  Discontinuous Galerkin Methods. H. T. Huynh. AIAA Paper 2007-4079
  <http://arc.aiaa.org/doi/abs/10.2514/6.2007-4079>`_

Linear Stability
----------------

The linear stability of an FR schemes depends on the form of the
correction function. Linear stability issues are discussed in:

- `A New Class of High-Order Energy Stable Flux Reconstruction Schemes.
  P. E. Vincent, P. Castonguay, A. Jameson. Journal of Scientific Computing,
  Volume 47, Number 1, Pages 50-72, 2011
  <http://www.springerlink.com/content/832853u112038372>`_
- `Insights from von Neumann Analysis of High-Order Flux Reconstruction
  Schemes. P. E. Vincent, P. Castonguay, A. Jameson. Journal of Computational
  Physics, Volume 230, Issue 22, Pages 8134-8154, 2011
  <http://www.sciencedirect.com/science/article/pii/S0021999111004323>`_ 
- `A New Class of High-Order Energy Stable Flux Reconstruction Schemes for
  Triangular Elements. P. Castonguay, P. E. Vincent, A. Jameson. Journal of
  Scientific Computing, Volume 51, Number 1, Pages 224-256, 2012
  <http://www.springerlink.com/content/u4514w1487786995/>`_ 
- `Energy Stable Flux Reconstruction Schemes for Advection-Diffusion Problems.
  P. Castonguay, D. M. Williams, P. E. Vincent, A. Jameson. Computer Methods
  in Applied Mechanics and Engineering, Volume 267, Pages 400-417, 2013
  <http://www.sciencedirect.com/science/article/pii/S0045782513002156>`_ 
- `Energy Stable Flux Reconstruction Schemes for Advection-Diffusion Problems
  on Triangles. D. M. Williams, P. Castonguay, P. E. Vincent, A. Jameson.
  Journal of Computational Physics, Volume 250, Pages 53-76, 2013
  <http://www.sciencedirect.com/science/article/pii/S0021999113003318>`_
- `Energy Stable Flux Reconstruction Schemes for Advection-Diffusion 
  Problems on Tetrahedra. D. M. Williams, A. Jameson. Journal of 
  Scientific Computing, Volume 59, Pages 721-759, 2014
  <http://link.springer.com/article/10.1007%2Fs10915-013-9780-2>`_
- `An Extended Range of Stable-Symmetric-Conservative Flux Reconstruction
  Correction Functions. P. E. Vincent, A. M. Farrington, F. D. Witherden,
  A. Jameson. Computer Methods in Applied Mechanics and Engineering,
  Volume 296, Pages 248-272, 2015 
  <http://www.sciencedirect.com/science/article/pii/S0045782515002418>`_

Non-Linear Stability
--------------------

The non-linear stability of an FR schemes depends on the location of the
solution points. Non-linear stability issues are discussed in:

- `On the Non-Linear Stability of Flux Reconstruction Schemes. A. Jameson,
  P. E. Vincent, P. Castonguay. Journal of Scientific Computing, Volume 50,
  Number 2, Pages 434-445, 2012
  <http://www.springerlink.com/content/n835050u01257r36>`_ 
- `An Analysis of Solution Point Coordinates for Flux Reconstruction Schemes on
  Triangular Elements. F. D. Witherden, P. E. Vincent. Journal of Scientific 
  Computing, Volume 61, Pages 398-423, 2014
  <http://link.springer.com/article/10.1007/s10915-014-9832-2>`_ 
