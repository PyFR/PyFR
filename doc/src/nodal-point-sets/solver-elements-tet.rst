*************************************
[solver-elements-tet{-mg-p\ *order*}]
*************************************

Parameterises the tetrahedral elements, or if -mg-p\ *order* is suffixed
the tetrahedral elements at multi-p level *order*, with

#. ``soln-pts`` --- location of the solution points in a tetrahedral
   element:

    ``alpha-opt`` | ``shunn-ham``

#. ``quad-deg`` --- degree of quadrature rule for anti-aliasing in a
   tetrahedral element:

    *int*

#. ``quad-npts`` --- number of points of the quadrature rule for
   anti-aliasing in a tetrahedral element:

    *int*

#. ``quad-pts`` --- name of quadrature rule for anti-aliasing in a
   tetrahedral element:

    ``shunn-ham`` | ``witherden`` | ``witherden-vincent``

Example::

    [solver-elements-tet]
    soln-pts = shunn-ham
    quad-deg = 9
    quad-pts = shunn-ham
