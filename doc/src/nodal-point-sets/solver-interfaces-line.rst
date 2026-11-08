************************
[solver-interfaces-line]
************************

Parameterises the line interfaces, with

#. ``flux-pts`` --- location of the flux points on a line interface:

    ``gauss-legendre`` | ``gauss-legendre-lobatto``

#. ``anti-alias`` --- whether to anti-alias the common flux on a
   line interface, in which case the quadrature points below
   take the place of the flux points:

    ``true`` | ``false``

#. ``quad-deg`` --- degree of quadrature rule for anti-aliasing on a
   line interface:

    *int*

#. ``quad-npts`` --- number of points of the quadrature rule for
   anti-aliasing on a line interface:

    *int*

#. ``quad-pts`` --- name of quadrature rule for anti-aliasing on a
   line interface:

    ``gauss-legendre`` | ``gauss-legendre-lobatto``

Example:

.. code-block:: ini

    [solver-interfaces-line]
    flux-pts = gauss-legendre
    quad-deg = 10
    quad-pts = gauss-legendre
