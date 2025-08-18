*************************************
[solver-elements-pri{-mg-p\ *order*}]
*************************************

Parameterises the prismatic elements, or if -mg-p\ *order* is suffixed
the prismatic elements at multi-p level *order*, with

#. ``soln-pts`` --- location of the solution points in a prismatic
   element:

    ``alpha-opt~gauss-legendre-lobatto`` |
    ``williams-shunn~gauss-legendre`` |
    ``williams-shunn~gauss-legendre-lobatto``

#. ``quad-deg`` --- degree of quadrature rule for anti-aliasing in a
   prismatic element:

    *int*

#. ``quad-npts`` --- number of points of the quadrature rule for
   anti-aliasing in a prismatic element:

    *int*

#. ``quad-pts`` --- name of quadrature rule for anti-aliasing in a
   prismatic element:

    ``williams-shunn~gauss-legendre`` |
    ``williams-shunn~gauss-legendre-lobatto`` | ``witherden-vincent``

Example::

    [solver-elements-pri]
    soln-pts = williams-shunn~gauss-legendre
    quad-deg = 10
    quad-pts = williams-shunn~gauss-legendre
