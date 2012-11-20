======
Theory
======

Introduction
------------

Theoretical studies and numerical experiments suggest that high-order
methods for unstructured grids can provide solutions to otherwise
intractable fluid flow problems within complex geometries. However,
despite their potential benefits, the use of unstructured high-order
methods remains limited in both academia and industry. There are
various reasons for this situation. These include difficulties
generating unstructured high-order meshes, problems resolving 
discontinuous solutions such as shock waves, and the relative 
complexity of unstructured high-order methods (compared with low-order 
schemes) [VJ2011]_.

Flux Reconstruction
------------

In an effort to address the latter issue of complexity, Huynh proposed 
the flux reconstruction (FR) approach to high-order methods 
[AAAA]_, which has been extended to triangular elements and 
used to solve the two-dimensional (2D) Euler equations by Wang and 
Gao [AAAA]_. The FR approach provides a simple and intuitive 
framework within which various well known high-order schemes can be 
cast. In particular, using the FR approach one can recover both 
collocation based nodal discontinuous Galerkin (DG) methods of the 
type described by Hesthaven and Warburton [AAAA]_, and 
spectral difference (SD) methods (at least for a linear flux 
function), which were originally proposed by Kopriva and Kolias 
[AAAA]_, and later generalized by Liu, 
Vinokur and Wang [AAAA]_.

In addition to recovering known methods, the FR approach also 
facilitates the definition of new schemes. Huynh previously identified 
several new unstructured high-order methods using the FR approach 
[H2007]_[H2009]_. Of particular note is the so called $g_2$ 
method, which Huynh showed to be stable for various orders of accuracy 
(using von Neumann analysis), and which can be made more efficient 
than other FR methods via judicious placement of the solution points 
\cite{huynh07}. Additionally, Huynh proposed various guidelines for 
selecting the so called flux correction function, which determines 
numerous properties of the associated FR scheme. In particular, for 
one-dimensional (1D) FR schemes, Huynh suggested (based on von Neumann 
analysis) that if a flux correction function of degree $k+1$ is 
orthogonal to all polynomials of degree $k-2$ then the resulting 
scheme will be linearly stable. Recently, Vincent, Castonguay and 
Jameson proved this assertion to be true using an energy method 
[VCJ2011A]_, and consequently identified a range of FR schemes 
(parameterized by a single scalar), which are guaranteed to be 
linearly stable for all orders of accuracy. These linearly stable FR 
schemes will henceforth be referred to as 
Vincent-Castonguay-Jameson-Huynh (VCJH) schemes.

.. [CVJ2012] A New Class of High-Order Energy Stable Flux
   Reconstruction Schemes for Triangular Elements P. Castonguay,
   P. E. Vincent, A. Jameson. Journal of Scientific Computing, Volume 
   51, Number 1, April 2012, Pages 224-256.

.. [JVC2012] On the Non-Linear Stability of Flux Reconstruction Schemes 
   A. Jameson, P. E. Vincent, P. Castonguay. Journal of Scientific 
   Computing, Volume 50, Number 2, February 2012, Pages 434-445.

.. [VCJ2011B] Insights from von Neumann Analysis of High-Order Flux 
   Reconstruction Schemes P. E. Vincent, P. Castonguay, A. Jameson. 
   Journal of Computational Physics, Volume 230, Issue 22, 2011, Pages 
   8134-8154.

.. [VJ2011] Facilitating the Adoption of Unstructured High-Order 
   Methods Amongst a Wider Community of Fluid Dynamicists P. E. 
   Vincent, A. Jameson. Mathematical Modelling of Natural Phenomena, 
   Volume 6, Issue 3, 2011, Pages 97-140.

.. [VCJ2011A] A New Class of High-Order Energy Stable Flux 
   Reconstruction Schemes P. E. Vincent, P. Castonguay, A. Jameson. 
   Journal of Scientific Computing, Volume 47, Number 1, April 2011, 
   Pages 50-72.

