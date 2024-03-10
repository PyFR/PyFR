*************************************
[solver-elements-pyr{-mg-p\ *order*}]
*************************************

Parameterises the pyramidal elements, or if -mg-p\ *order* is suffixed
the pyramidal elements at multi-p level *order*, with

1. ``soln-pts`` --- location of the solution points in a pyramidal
   element:

    ``gauss-legendre`` | ``gauss-legendre-lobatto``

2. ``quad-deg`` --- degree of quadrature rule for anti-aliasing in a
   pyramidal element:

    *int*

3. ``quad-pts`` --- name of quadrature rule for anti-aliasing in a
   pyramidal element:

    ``witherden-vincent``

Example::

    [solver-elements-pyr]
    soln-pts = gauss-legendre
    quad-deg = 10
    quad-pts = witherden-vincent
