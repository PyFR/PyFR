*********************
[solver-elements-tet]
*********************

Parameterises the tetrahedral elements, with

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

Example:

.. code-block:: ini

    [solver-elements-tet]
    soln-pts = shunn-ham
    quad-deg = 9
    quad-pts = shunn-ham
