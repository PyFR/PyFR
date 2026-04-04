*********************
[solver-elements-tri]
*********************

Parameterises the triangular elements, with

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

Example:

.. code-block:: ini

    [solver-elements-tri]
    soln-pts = williams-shunn
    quad-deg = 10
    quad-pts = williams-shunn
