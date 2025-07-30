***************************************
[solver-interfaces-tri{-mg-p\ *order*}]
***************************************

Parameterises the triangular interfaces, or if -mg-p\ *order* is
suffixed the triangular interfaces at multi-p level *order*, with

#. ``flux-pts`` --- location of the flux points on a triangular
   interface:

    ``alpha-opt`` | ``williams-shunn``

#. ``quad-deg`` --- degree of quadrature rule for anti-aliasing on a
   triangular interface:

    *int*

#. ``quad-npts`` --- number of points of the quadrature rule for
   anti-aliasing on a triangular interface:

    *int*

#. ``quad-pts`` --- name of quadrature rule for anti-aliasing on a
   triangular interface:

    ``williams-shunn`` | ``witherden-vincent``

Example::

    [solver-interfaces-tri]
    flux-pts = williams-shunn
    quad-deg = 10
    quad-pts = williams-shunn
