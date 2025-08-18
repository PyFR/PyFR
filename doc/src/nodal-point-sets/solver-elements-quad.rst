**************************************
[solver-elements-quad{-mg-p\ *order*}]
**************************************

Parameterises the quadrilateral elements, or if -mg-p\ *order* is
suffixed the quadrilateral elements at multi-p level *order*, with

#. ``soln-pts`` --- location of the solution points in a quadrilateral
   element:

    ``gauss-legendre`` | ``gauss-legendre-lobatto``

#. ``quad-deg`` --- degree of quadrature rule for anti-aliasing in a
   quadrilateral element:

    *int*

#. ``quad-npts`` --- number of points of the quadrature rule for
   anti-aliasing in a quadrilateral element:

    *int*

#. ``quad-pts`` --- name of quadrature rule for anti-aliasing in a
   quadrilateral element:

    ``gauss-legendre`` | ``gauss-legendre-lobatto`` |
    ``witherden-vincent``

Example::

    [solver-elements-quad]
    soln-pts = gauss-legendre
    quad-deg = 10
    quad-pts = gauss-legendre
