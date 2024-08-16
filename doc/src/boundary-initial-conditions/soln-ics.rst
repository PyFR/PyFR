**********
[soln-ics]
**********

Parameterises space (x, y, [z]) dependent initial conditions with

1. ``rho`` --- initial density distribution for ``euler`` |
   ``navier-stokes``:

    *string*

2. ``u`` --- initial x-velocity distribution for ``euler`` |
   ``navier-stokes`` | ``ac-euler`` | ``ac-navier-stokes``:

    *string*

3. ``v`` --- initial y-velocity distribution for ``euler`` |
   ``navier-stokes`` | ``ac-euler`` | ``ac-navier-stokes``:

    *string*

4. ``w`` --- initial z-velocity distribution for ``euler`` |
   ``navier-stokes`` | ``ac-euler`` | ``ac-navier-stokes``:

    *string*

5. ``p`` --- initial static pressure distribution for ``euler`` |
   ``navier-stokes`` | ``ac-euler`` | ``ac-navier-stokes``:

    *string*

6. ``quad-deg-{etype}`` --- degree of quadrature rule to perform L2 projection (optional):

    *int*

7. ``quad-pts-{etype}`` --- name of quadrature rule to perform L2 projection (optional):

    *string*


Example::

    [soln-ics]
    rho = 1.0
    u = x*y*sin(y)
    v = z
    w = 1.0
    p = 1.0/(1.0+x)
    quad-deg = 9
    quad-pts-hex = gauss-legendre
