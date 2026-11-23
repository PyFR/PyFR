*********************
[solver-elements-pyr]
*********************

Parameterises the pyramidal elements, with

#. ``soln-pts`` --- location of the solution points in a pyramidal
   element:

    ``alpha-opt`` | ``gauss-legendre`` | ``gauss-legendre-lobatto``

#. ``anti-alias`` --- whether to anti-alias the flux in a pyramidal
   element with the quadrature rule below:

    ``true`` | ``false``

#. ``quad-deg`` --- degree of quadrature rule for anti-aliasing in a
   pyramidal element:

    *int*

#. ``quad-npts`` --- number of points of the quadrature rule for
   anti-aliasing in a pyramidal element:

    *int*

#. ``quad-pts`` --- name of quadrature rule for anti-aliasing in a
   pyramidal element:

    ``gauss-legendre~gauss-jacobi``

#. ``operators`` --- space in which a pyramidal element is
   discretised:

    ``physical``

Example:

.. code-block:: ini

    [solver-elements-pyr]
    soln-pts = gauss-legendre
    quad-deg = 8
    quad-pts = gauss-legendre~gauss-jacobi
