*************************************
[solver-elements-tet{-mg-p\ *order*}]
*************************************

Parameterises the tetrahedral elements, or if -mg-p\ *order* is suffixed
the tetrahedral elements at multi-p level *order*, with

1. ``soln-pts`` --- location of the solution points in a tetrahedral
   element:

    ``alpha-opt`` | ``shunn-ham``

2. ``quad-deg`` --- degree of quadrature rule for anti-aliasing in a
   tetrahedral element:

    *int*

3. ``quad-pts`` --- name of quadrature rule for anti-aliasing in a
   tetrahedral element:

    ``shunn-ham`` | ``witherden-vincent``

Example::

    [solver-elements-tet]
    soln-pts = shunn-ham
    quad-deg = 10
    quad-pts = shunn-ham
