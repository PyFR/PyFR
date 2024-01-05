**************************
[solver-plugin-turbulence]
**************************

Injects synthetic eddies into a region of the domain. Parameterised with

1. ``avg-rho`` --- average free-stream density:

    *float*

2. ``avg-u`` --- average free-stream velocity magnitude:

    *float*

3. ``avg-mach`` --- averge free-stream Mach number:

    *float*

4. ``turbulence-intensity`` --- percentage turbulence intensity:

    *float*

5. ``turbulence-length-scale`` --- turbulent length scale:

    *float*

6. ``sigma`` --- standard deviation of Gaussian sythetic eddy profile:

    *float*

7. ``centre`` --- centre of plane on which synthetic eddies are injected:

    (*float*, *float*, *float*)

8. ``y-dim`` --- y-dimension of plane:

    *float*

9. ``z-dim`` --- z-dimension of plane:

    *float*

10. ``rot-axis`` --- axis about which plane is rotated:

    (*float*, *float*, *float*)

11. ``rot-angle`` --- angle in degrees that plane is rotated:

    *float*

Example::

    [solver-plugin-turbulence]
    avg-rho = 1.0
    avg-u = 1.0
    avg-mach = 0.2
    turbulence-intensity = 1.0
    turbulence-length-scale = 0.075
    sigma = 0.7
    centre = (0.15, 2.0, 2.0)
    y-dim = 3.0
    z-dim = 3.0
    rot-axis = (0, 0, 1)
    rot-angle = 0.0
