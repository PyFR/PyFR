************************
[solver-time-integrator]
************************

Parameterises the time-integration scheme used by the solver with

1. ``formulation`` --- formulation:

    ``std`` | ``dual``

    where

    ``std`` requires

        - ``scheme`` --- time-integration scheme

           ``euler`` | ``rk34`` | ``rk4`` | ``rk45`` | ``tvd-rk3``

        - ``tstart`` --- initial time

           *float*

        - ``tend`` --- final time

           *float*

        - ``dt`` --- time-step

           *float*

        - ``controller`` --- time-step controller

           ``none`` | ``pi``

           where

           ``pi`` only works with ``rk34`` and ``rk45`` and requires

            - ``atol`` --- absolute error tolerance

               *float*

            - ``rtol`` --- relative error tolerance

               *float*

            - ``errest-norm`` --- norm to use for estimating the error

               ``uniform`` | ``l2``

            - ``safety-fact`` --- safety factor for step size adjustment
              (suitable range 0.80-0.95)

               *float*

            - ``min-fact`` --- minimum factor by which the time-step can
              change between iterations (suitable range 0.1-0.5)

               *float*

            - ``max-fact`` --- maximum factor by which the time-step can
              change between iterations (suitable range 2.0-6.0)

               *float*

            - ``dt-max`` --- maximum permissible time-step

               *float*

    ``dual`` requires

        - ``scheme`` --- time-integration scheme

           ``backward-euler`` | ``sdirk33`` | ``sdirk43``

        - ``pseudo-scheme`` --- pseudo time-integration scheme

           ``euler`` | ``rk34`` | ``rk4`` | ``rk45`` | ``tvd-rk3`` | ``vermeire``

        - ``tstart`` --- initial time

           *float*

        - ``tend`` --- final time

           *float*

        - ``dt`` --- time-step

           *float*

        - ``controller`` --- time-step controller

           ``none``

        - ``pseudo-dt`` --- pseudo time-step

           *float*

        - ``pseudo-niters-max`` --- minimum number of iterations

           *int*

        - ``pseudo-niters-min`` --- maximum number of iterations

           *int*

        - ``pseudo-resid-tol`` --- pseudo residual tolerance

           *float*

        - ``pseudo-resid-norm`` --- pseudo residual norm

           ``uniform`` | ``l2``

        - ``pseudo-controller`` --- pseudo time-step controller

           ``none`` | ``local-pi``

           where

           ``local-pi`` only works with ``rk34`` and ``rk45`` and
           requires

            - ``atol`` --- absolute error tolerance

               *float*

            - ``safety-fact`` --- safety factor for pseudo time-step
              size adjustment (suitable range 0.80-0.95)

               *float*

            - ``min-fact`` --- minimum factor by which the local
              pseudo time-step can change between iterations
              (suitable range 0.98-0.998)

               *float*

            - ``max-fact`` --- maximum factor by which the local
              pseudo time-step can change between iterations
              (suitable range 1.001-1.01)

               *float*

            - ``pseudo-dt-min-mult`` --- minimum permissible
              local pseudo time-step given as a
              multiplier of ``pseudo-dt`` (suitable range 0.001-1.0)

               *float*

            - ``pseudo-dt-max-mult`` --- maximum permissible
              local pseudo time-step given as a
              multiplier of ``pseudo-dt`` (suitable range 2.0-5.0)

               *float*

2. ``dt-adjust-min-fact`` --- minimum allowed factor by which the 
   time-step modified by controller can be further changed to 
   satisfy the constraints set by the target time
   (suitable range 0.5-0.99)

    *float*

3. ``dt-adjust-max-fact`` --- maximum allowed factor by which the 
   time-step modified by controller can be further changed to 
   satisfy the constraints set by the target time
    (suitable range 1.0-1.1)

    *float*

Example::

    [solver-time-integrator]
    formulation = std
    scheme = rk45
    controller = pi
    tstart = 0.0
    tend = 10.0
    dt = 0.001
    atol = 0.00001
    rtol = 0.00001
    errest-norm = l2
    safety-fact = 0.9
    min-fact = 0.3
    max-fact = 2.5
    dt-adjust-min-fact = 0.99
    dt-adjust-max-fact = 1.01
