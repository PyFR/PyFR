*************************************
[solver-elements-pri{-mg-p\ *order*}]
*************************************

Parameterises the prismatic elements, or if -mg-p\ *order* is suffixed
the prismatic elements at multi-p level *order*, with

1. ``soln-pts`` --- location of the solution points in a prismatic
   element:

    ``alpha-opt~gauss-legendre-lobatto`` |
    ``williams-shunn~gauss-legendre`` |
    ``williams-shunn~gauss-legendre-lobatto``

2. ``quad-deg`` --- degree of quadrature rule for anti-aliasing in a
   prismatic element:

    *int*

3. ``quad-pts`` --- name of quadrature rule for anti-aliasing in a
   prismatic element:

    ``williams-shunn~gauss-legendre`` |
    ``williams-shunn~gauss-legendre-lobatto`` | ``witherden-vincent``

Example::

    [solver-elements-pri]
    soln-pts = williams-shunn~gauss-legendre
    quad-deg = 10
    quad-pts = williams-shunn~gauss-legendre
