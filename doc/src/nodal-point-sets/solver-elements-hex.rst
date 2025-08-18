*************************************
[solver-elements-hex{-mg-p\ *order*}]
*************************************

Parameterises the hexahedral elements, or if -mg-p\ *order* is suffixed
the hexahedral elements at multi-p level *order*, with

#. ``soln-pts`` --- location of the solution points in a hexahedral
   element:

    ``gauss-legendre`` | ``gauss-legendre-lobatto``

#. ``quad-deg`` --- degree of quadrature rule for anti-aliasing in a
   hexahedral element:

    *int*

#. ``quad-npts`` --- number of points of the quadrature rule for
   anti-aliasing in a hexahedral element:

    *int*

#. ``quad-pts`` --- name of quadrature rule for anti-aliasing in a
   hexahedral element:

    ``gauss-legendre`` | ``gauss-legendre-lobatto`` |
    ``witherden-vincent``

Example::

    [solver-elements-hex]
    soln-pts = gauss-legendre
    quad-npts = 216
    quad-pts = gauss-legendre
