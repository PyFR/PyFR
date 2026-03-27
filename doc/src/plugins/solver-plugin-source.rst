**********************
[solver-plugin-source]
**********************

Injects solution, space (x, y, [z]), and time (t) dependent
source terms with

#. ``rho`` --- density source term:

    *string*

#. ``rhou`` --- x-momentum source term:

    *string*

#. ``rhov`` --- y-momentum source term:

    *string*

#. ``rhow`` --- z-momentum source term:

    *string*

#. ``E`` --- energy source term:

    *string*

Example:

.. code-block:: ini

    [solver-plugin-source]
    rho = t
    rhou = x*y*sin(y)
    rhov = z*rho
    rhow = 1.0
    E = 1.0/(1.0+x)
