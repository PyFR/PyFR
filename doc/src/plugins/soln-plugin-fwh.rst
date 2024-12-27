*****************
[soln-plugin-fwh]
*****************

Use Ffowcs Williams--Hawkings equation to approximate far field noise in
a uniformly moving medium:

#. ``tstart`` --- time at which to start sampling, default is ``0``:

    *float*

#. ``dt`` --- time step between samples:

    *float*

#. ``file`` --- output file path; should the file already exist it
   will be appended to:

    *string*

#. ``file-header`` --- if to output a header row or not:

    *boolean*

#. ``surface`` --- a region the surface of which is saple for the FWH
   sovler, only use a combination of the geometric shapes specified in
   :ref:`regions`:

   ``shape(args, ...)``

#. ``quad-deg`` --- degree of surface quadrature rule (optional):

    *int*

#. ``quad-pts-{etype}`` --- name of surface quadrature rule (optional):

    *string*

#. ``observer-pts`` --- the obversation point in the far field at which
   noise is approximated:

   ``[(x, y), (x, y), ...]`` | ``[(x, y, z), (x, y, z), ...]``

#. ``rho, u, v, (w), p, (c)`` --- the constant far field properties of
   the flow. For incompressible calculations the sound speed ``c`` and
   the dnsity ``rho`` must be given:

    *float*

Example::

    [soln-plugin-fwh]
    file = fwh.csv
    file-header = true
    region = box((1, -5), (10, 5))
    tstart = 10
    dt = 1e-2
    observer-pts = [(1, 10), (1, 30), (1, 100), (1, 300)]

    rho = 1
    u = 1
    v = 0
    p = 10
