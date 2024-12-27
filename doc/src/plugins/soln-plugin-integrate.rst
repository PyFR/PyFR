.. _soln-plugin-integrate:

***********************
[soln-plugin-integrate]
***********************

Integrate quantities over the compuational domain. Parameterised with:

#. ``nsteps`` --- calculate the integral every ``nsteps`` time steps:

    *int*

#. ``file`` --- output file path; should the file already exist it will
   be appended to:

    *string*

#. ``file-header`` --- if to output a header row or not:

    *boolean*

#. ``quad-deg`` --- degree of quadrature rule (optional):

    *int*

#. ``quad-pts-{etype}`` --- name of quadrature rule (optional):

    *string*

#. ``norm`` --- sets the degree and calculates an :math:`L_p` norm,
    otherwise standard integration is performed:

    *float* | ``inf`` | ``none``

#. ``region`` --- region to integrate, specified as either the
   entire domain using ``*`` or a combination of the geometric shapes
   specified in :ref:`regions`:

    ``*`` | ``shape(args, ...)``

#. ``int``-*name* --- expression to integrate, written as a function of
   the primitive variables and gradients thereof, the physical
   coordinates [x, y, [z]] and/or the physical time [t]; multiple
   expressions, each with their own *name*, may be specified:

    *string*

Example::

    [soln-plugin-integrate]
    nsteps = 50
    file = integral.csv
    file-header = true
    quad-deg = 9
    vor1 = (grad_w_y - grad_v_z)
    vor2 = (grad_u_z - grad_w_x)
    vor3 = (grad_v_x - grad_u_y)

    int-E = rho*(u*u + v*v + w*w)
    int-enst = rho*(%(vor1)s*%(vor1)s + %(vor2)s*%(vor2)s + %(vor3)s*%(vor3)s)
