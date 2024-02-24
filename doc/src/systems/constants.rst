***********
[constants]
***********

Sets constants used in the simulation

1. ``gamma`` --- ratio of specific heats for ``euler`` |
   ``navier-stokes``:

    *float*

2. ``mu`` --- dynamic viscosity for ``navier-stokes``:

    *float*

3. ``nu`` --- kinematic viscosity for ``ac-navier-stokes``:

    *float*

4. ``Pr`` --- Prandtl number for ``navier-stokes``:

    *float*

5. ``cpTref`` --- product of specific heat at constant pressure and
   reference temperature for ``navier-stokes`` with Sutherland's Law:

   *float*

6. ``cpTs`` --- product of specific heat at constant pressure and
   Sutherland temperature for ``navier-stokes`` with Sutherland's Law:

   *float*

7. ``ac-zeta`` --- artificial compressibility factor for ``ac-euler`` |
   ``ac-navier-stokes``

   *float*

Other constant may be set by the user which can then be used throughout the
``.ini`` file.

Example::

    [constants]
    ; PyFR Constants
    gamma = 1.4
    mu = 0.001
    Pr = 0.72

    ; User Defined Constants
    V_in = 1.0
    P_out = 20.0
