***********************
Post-processing plugins
***********************

Post-processing plugins compute derived fields during ``pyfr export``
and write them to the output VTK file as additional point data.
Plugins are activated with the ``--postproc name`` flag, which may be
repeated to apply multiple plugins.

Plugins that require parameters read them from sections named
``[postproc-plugin-name]``.  By default these are read from the
solution file's embedded config; an alternative INI file may be
provided via ``--cfg``.

Example:

.. code-block:: shell

    pyfr export volume --postproc mach mesh.pyfrm soln.pyfrs out.vtu
    pyfr export boundary --postproc cf --postproc yplus --cfg pp.ini \
        mesh.pyfrm soln.pyfrs out.vtu wall

Plugins that require gradient data only work with solution files that
were written with ``write-gradients = true`` in the
``[soln-plugin-writer]`` section.

[postproc-plugin-mach]
======================

Mach number for compressible Euler and Navier-Stokes systems.  No
parameters required.

[postproc-plugin-isen-mach]
===========================

Isentropic Mach number assuming a known total pressure.

#. ``p-total`` --- total (stagnation) pressure:

    *float*

[postproc-plugin-cp]
====================

Pressure coefficient :math:`(p - p_\infty) / (\frac{1}{2}\rho_\infty
u_\infty^2)`.

#. ``rho-inf`` --- freestream density:

    *float*

#. ``u-inf`` --- freestream velocity magnitude:

    *float*

#. ``p-inf`` --- freestream static pressure:

    *float*

[postproc-plugin-cf]
====================

Skin friction coefficient :math:`\tau_w / (\frac{1}{2}\rho_\infty
u_\infty^2)`.  Boundary export only.  Requires gradient data.

#. ``rho-inf`` --- freestream density:

    *float*

#. ``u-inf`` --- freestream velocity magnitude:

    *float*

[postproc-plugin-vorticity]
===========================

Vorticity vector (3D) or scalar (2D).  Requires gradient data.  No
parameters required.

[postproc-plugin-yplus]
=======================

Wall :math:`y^+` based on the distance from the wall to the nearest
interior solution point.  Boundary export only.  Requires gradient
data.  No parameters required.

.. note::

   The wall distance is approximated as the physical distance from
   the wall face to the nearest interior solution point of the boundary
   element.
