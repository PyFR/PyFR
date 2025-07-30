*************************************
[solver-elements-tri{-mg-p\ *order*}]
*************************************

Parameterises the triangular elements, or if -mg-p\ *order* is suffixed
the triangular elements at multi-p level *order*, with

#. ``soln-pts`` --- location of the solution points in a triangular
   element:

    ``alpha-opt`` | ``williams-shunn``

#. ``quad-deg`` --- degree of quadrature rule for anti-aliasing in a
   triangular element:

    *int*

#. ``quad-npts`` --- number of points of the quadrature rule for
   anti-aliasing in a triangular element:

    *int*

#. ``quad-pts`` --- name of quadrature rule for anti-aliasing in a
   triangular element:

    ``williams-shunn`` | ``witherden-vincent``

Example::

    [solver-elements-tri]
    soln-pts = williams-shunn
    quad-deg = 10
    quad-pts = williams-shunn
