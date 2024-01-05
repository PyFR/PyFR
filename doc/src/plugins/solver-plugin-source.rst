**********************
[solver-plugin-source]
**********************

Injects solution, space (x, y, [z]), and time (t) dependent
source terms with

1. ``rho`` --- density source term for ``euler`` | ``navier-stokes``:

    *string*

2. ``rhou`` --- x-momentum source term for ``euler`` | ``navier-stokes``
   :

    *string*

3. ``rhov`` --- y-momentum source term for ``euler`` | ``navier-stokes``
   :

    *string*

4. ``rhow`` --- z-momentum source term for ``euler`` | ``navier-stokes``
   :

    *string*

5. ``E`` --- energy source term for ``euler`` | ``navier-stokes``
   :

    *string*

6. ``p`` --- pressure source term for ``ac-euler`` |
   ``ac-navier-stokes``:

    *string*

7. ``u`` --- x-velocity source term for ``ac-euler`` |
   ``ac-navier-stokes``:

    *string*

8. ``v`` --- y-velocity source term for ``ac-euler`` |
   ``ac-navier-stokes``:

    *string*

9. ``w`` --- w-velocity source term for ``ac-euler`` |
   ``ac-navier-stokes``:

    *string*

Example::

    [solver-plugin-source]
    rho = t
    rhou = x*y*sin(y)
    rhov = z*rho
    rhow = 1.0
    E = 1.0/(1.0+x)
