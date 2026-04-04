**********
[soln-ics]
**********

Parameterises space (x, y, [z]) dependent initial conditions with

#. ``rho`` --- initial density distribution:

    *string*

#. ``u`` --- initial x-velocity distribution:

    *string*

#. ``v`` --- initial y-velocity distribution:

    *string*

#. ``w`` --- initial z-velocity distribution:

    *string*

#. ``p`` --- initial static pressure distribution:

    *string*

6. ``quad-deg-{etype}`` --- degree of quadrature rule to perform L2
   projection (optional):

    *int*

7. ``quad-pts-{etype}`` --- name of quadrature rule to perform L2
   projection (optional):

    *string*


Example:

.. code-block:: ini

    [soln-ics]
    rho = 1.0
    u = x*y*sin(y)
    v = z
    w = 1.0
    p = 1.0/(1.0+x)
    quad-deg = 9
    quad-pts-hex = gauss-legendre
