*************************************
[solver-elements-pyr{-mg-p\ *order*}]
*************************************

Parameterises the pyramidal elements, or if -mg-p\ *order* is suffixed
the pyramidal elements at multi-p level *order*, with

#. ``soln-pts`` --- location of the solution points in a pyramidal
   element:

    ``gauss-legendre`` | ``gauss-legendre-lobatto``

#. ``quad-deg`` --- degree of quadrature rule for anti-aliasing in a
   pyramidal element:

    *int*

#. ``quad-npts`` --- number of points of the quadrature rule for
   anti-aliasing in a pyramidal element:

    *int*

#. ``quad-pts`` --- name of quadrature rule for anti-aliasing in a
   pyramidal element:

    ``witherden`` | ``witherden-vincent``

Example::

    [solver-elements-pyr]
    soln-pts = gauss-legendre
    quad-deg = 10
    quad-pts = witherden-vincent
