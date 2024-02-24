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

Example::

    [soln-ics]
    rho = 1.0
    u = x*y*sin(y)
    v = z
    w = 1.0
    p = 1.0/(1.0+x)
