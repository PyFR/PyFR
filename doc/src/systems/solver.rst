********
[solver]
********

Parameterises the solver with

1. ``system`` --- governing system:

    ``euler`` | ``navier-stokes`` | ``ac-euler`` | ``ac-navier-stokes``

    where

    ``euler`` requires

        - ``shock-capturing`` --- shock capturing scheme:

          ``none`` | ``entropy-filter``

    ``navier-stokes`` requires

        - ``viscosity-correction`` --- viscosity correction:

          ``none`` | ``sutherland``

        - ``shock-capturing`` --- shock capturing scheme:

          ``none`` | ``artificial-viscosity`` | ``entropy-filter``

2. ``order`` --- order of polynomial solution basis:

    *int*

3. ``anti-alias`` --- type of anti-aliasing:

    ``flux`` | ``surf-flux`` | ``flux, surf-flux``

Example::

    [solver]
    system = navier-stokes
    order = 3
    anti-alias = flux
    viscosity-correction = none
    shock-capturing = none
