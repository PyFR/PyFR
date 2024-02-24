****************************************
[solver-interfaces-line{-mg-p\ *order*}]
****************************************

Parameterises the line interfaces, or if -mg-p\ *order* is suffixed the
line interfaces at multi-p level *order*, with

1. ``flux-pts`` --- location of the flux points on a line interface:

    ``gauss-legendre`` | ``gauss-legendre-lobatto``

2. ``quad-deg`` --- degree of quadrature rule for anti-aliasing on a
   line interface:

    *int*

3. ``quad-pts`` --- name of quadrature rule for anti-aliasing on a
   line interface:

    ``gauss-legendre`` | ``gauss-legendre-lobatto``

Example::

    [solver-interfaces-line]
    flux-pts = gauss-legendre
    quad-deg = 10
    quad-pts = gauss-legendre
