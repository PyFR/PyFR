***********
[constants]
***********

Sets constants used in the simulation

#. ``gamma`` --- ratio of specific heats:

    *float*

#. ``mu`` --- dynamic viscosity for ``navier-stokes``:

    *float*

#. ``Pr`` --- Prandtl number for ``navier-stokes``:

    *float*

#. ``cpTref`` --- product of specific heat at constant pressure and
   reference temperature for ``navier-stokes`` with Sutherland's Law:

   *float*

#. ``cpTs`` --- product of specific heat at constant pressure and
   Sutherland temperature for ``navier-stokes`` with Sutherland's Law:

   *float*

#. ``mach-ref`` --- lower bound on the local Mach number by which the
   ``roe-lm`` Riemann solver scales its acoustic dissipation; the
   solver is low Mach number consistent only above this bound, a value
   of zero gives the original scheme of Rieper and a value of one or
   more recovers ``roe``:

   *float* (default: 0.01)

Other constant may be set by the user which can then be used throughout the
``.ini`` file.

Example:

.. code-block:: ini

    [constants]
    ; PyFR Constants
    gamma = 1.4
    mu = 0.001
    Pr = 0.72

    ; User Defined Constants
    V_in = 1.0
    P_out = 20.0
