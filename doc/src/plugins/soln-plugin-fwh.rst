*****************
[soln-plugin-fwh]
*****************

Use Ffowcs Williams--Hawkings equation to approximate far field noise in a
uniformly moving medium:

1. ``tstart`` --- time at which to start sampling, default is ``0``:

    *float*

2. ``dt`` --- time step between samples:

    *float*

3. ``file`` --- output file path; should the file already exist it
   will be appended to:

    *string*

4. ``header`` --- if to output a header row or not:

    *boolean*

5. ``surface`` --- a region the surface of which is saple for the FWH sovler,
   only use a combination of the geometric shapes specified in :ref:`regions`:

   ``shape(args, ...)``

6. ``quad-deg`` --- degree of surface quadrature rule (optional):

    *int*

7. ``quad-pts-{etype}`` --- name of surface quadrature rule (optional):

    *string*

8. ``observer-pts`` --- the obversation point in the far field at which noise is
   approximated:

   ``[(x, y), (x, y), ...]`` | ``[(x, y, z), (x, y, z), ...]``

9. ``rho, u, v, (w), p, (c)`` --- the constant far field properties of the
   flow. For incompressible calculations the sound speed ``c`` and the dnsity
   ``rho`` must be given:

    *float*

Example::

    [soln-plugin-fwh]
    file = fwh.csv
    region = box((1, -5), (10, 5))
    header = true
    tstart = 10
    dt = 1e-2
    observer-pts = [(1, 10), (1, 30), (1, 100), (1, 300)]

    rho = 1
    u = 1
    v = 0
    p = 10
