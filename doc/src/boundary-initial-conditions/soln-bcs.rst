*****************
[soln-bcs-*name*]
*****************

Parameterises constant, or if available space (x, y, [z]) and time (t)
dependent, boundary condition labelled *name* in the .pyfrm file with

1. ``type`` --- type of boundary condition:

    ``ac-char-riem-inv`` | ``ac-in-fv`` | ``ac-out-fp`` | ``char-riem-inv`` |
    ``char-riem-inv-mass-flow`` | ``no-slp-adia-wall`` |
    ``no-slp-isot-wall`` | ``no-slp-wall`` | ``slp-adia-wall`` | ``slp-wall`` |
    ``sub-in-frv`` | ``sub-in-ftpttang`` | ``sub-out-fp`` | ``sup-in-fa`` |
    ``sup-out-fn``

    where

    ``char-riem-inv`` requires

        - ``rho`` --- density

           *float* | *string*

        - ``u`` --- x-velocity

           *float* | *string*

        - ``v`` --- y-velocity

           *float* | *string*

        - ``w`` --- z-velocity

           *float* | *string*

        - ``p`` --- static pressure

           *float* | *string*

    ``char-riem-inv-mass-flow`` requires

        - ``rho`` --- density

           *float* | *string*

        - ``u`` --- x-velocity

           *float* | *string*

        - ``v`` --- y-velocity

           *float* | *string*

        - ``w`` --- z-velocity

           *float* | *string*

        - ``p`` --- initial static pressure, the controller will vary this to
          target a mass flow rate.

           *float* | *string*

        - ``mass-flow-rate`` --- target mass flow rate across the boundary.

           *float* | *string*

        - ``alpha`` --- parameter between 0 and 1 for the exponentially
          weighted moving average of the mass flow rate.

           *float* | *string*

        - ``eta`` --- parameter greater than 0 controlling the strength of the
          controller. The appropriate strength is problem specific and varies
          depending on if the simulation has been nondimensionalised.

           *float* | *string*

        - ``nsteps`` --- number of Runge-Kutta steps between activations of the
          controller. Typically between 10 and 500.

           *int*

        - ``tstart`` --- start time of the mass flow controller, before this
          time the Riemann invariant remains fixed.

           *float*

        - ``quad-deg-{etype}`` --- degree of quadrature rule for mass flow
          integration (optional).

           *int*

        - ``quad-pts-{etype}`` --- name of quadrature rule (optional).

           *string*

        - ``file`` --- name of a CSV file to output statistics to (optional).

           *string*

        - ``flushsteps`` --- frequency to flush output to the CSV file
          (optional).

           *int*

    ``no-slp-adia-wall`` has no parameters

    ``no-slp-isot-wall`` requires

        - ``u`` --- x-velocity of wall

           *float*

        - ``v`` --- y-velocity of wall

           *float*

        - ``w`` --- z-velocity of wall

           *float*

        - ``cpTw`` --- product of specific heat capacity at constant
          pressure and temperature of wall

           *float*

    ``slp-adia-wall`` has no parameters

    ``sub-in-frv`` only works with ``navier-stokes`` and
    requires

        - ``rho`` --- density

           *float* | *string*

        - ``u`` --- x-velocity

           *float* | *string*

        - ``v`` --- y-velocity

           *float* | *string*

        - ``w`` --- z-velocity

           *float* | *string*

    ``sub-in-ftpttang`` only works with ``navier-stokes``
    and requires

        - ``pt`` --- total pressure

           *float*

        - ``cpTt`` --- product of specific heat capacity at constant
          pressure and total temperature

           *float*

        - ``theta`` --- azimuth angle (in degrees) of inflow measured
          in the x-y plane relative to the positive x-axis

           *float*

        - ``phi`` --- inclination angle (in degrees) of inflow measured
          relative to the positive z-axis

           *float*

    ``sub-out-fp`` only works with ``navier-stokes`` and
    requires

        - ``p`` --- static pressure

           *float* | *string*

    ``sup-in-fa`` requires

        - ``rho`` --- density

           *float* | *string*

        - ``u`` --- x-velocity

           *float* | *string*

        - ``v`` --- y-velocity

           *float* | *string*

        - ``w`` --- z-velocity

           *float* | *string*

        - ``p`` --- static pressure

           *float* | *string*

    ``sup-out-fn`` has no parameters

Example:

.. code-block:: ini

    [soln-bcs-bcwallupper]
    type = no-slp-isot-wall
    cpTw = 10.0
    u = 1.0

Simple periodic boundary conditions are supported; however, their
behaviour is not controlled through the ``.ini`` file, instead it is
handled at the mesh generation stage. Two faces may be tagged with
``periodic_x_l`` and ``periodic_x_r``, where ``x`` is a unique
identifier for the pair of boundaries. Currently, only periodicity in a
single cardinal direction is supported, for example, the planes
``(x,y,0)`` and ``(x,y,10)``.
