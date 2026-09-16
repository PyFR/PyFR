***********************
Post-processing plugins
***********************

Post-processing plugins compute derived fields during ``pyfr export``
and write them to the output VTK file as additional point data.
Plugins are activated with the ``--postproc name`` flag, which may be
repeated to apply multiple plugins.

Plugin inputs are read from either the ``[constants]`` section or a
``[postproc-plugin-name]`` section of the solution config (or a
``--cfg`` override).

Example:

.. code-block:: shell

    pyfr export volume --postproc mach mesh.pyfrm soln.pyfrs out.vtu
    pyfr export boundary --postproc cf --postproc yplus --cfg pp.ini \
        mesh.pyfrm soln.pyfrs out.vtu wall

Plugins that require gradient data only work with solution files that
were written with ``write-gradients = true`` in the
``[soln-plugin-writer]`` section.

mach
====

Mach number for compressible Euler and Navier-Stokes systems.  No
parameters required.

isen-mach
=========

Isentropic Mach number assuming a known total pressure.  Reads
``p-total`` from ``[constants]``.

cp
==

Pressure coefficient :math:`(p - p_\infty) / (\frac{1}{2}\rho_\infty
u_\infty^2)`.  Reads ``rho-inf``, ``u-inf``, ``p-inf`` from
``[constants]``.

cf
==

Skin friction coefficient :math:`\tau_w / (\frac{1}{2}\rho_\infty
u_\infty^2)`.  Boundary export only.  Requires gradient data.  Reads
``rho-inf`` and ``u-inf`` from ``[constants]``.

vorticity
=========

Vorticity vector (3D) or scalar (2D) for Navier-Stokes systems.
Requires gradient data.  No parameters required.

yplus
=====

Wall :math:`y^+` based on the distance from the wall to the nearest
interior solution point.  Boundary export only.  Requires gradient
data.  No parameters required.

.. note::

   The wall distance is approximated as the physical distance from
   the wall face to the nearest interior solution point of the boundary
   element.

nirf
====

Transforms the solution of a non-inertial reference frame run (see
:doc:`solver-plugin-nirf`) from the body frame back into the inertial
(lab) frame: the coordinates are rotated and, optionally, translated to
the body's lab-frame position, and the velocity, its gradients and any
derived quantities take their lab-frame values.  Reads ``center-of-rot``
from ``[solver-plugin-nirf]`` and, from ``[postproc-plugin-nirf]``:

#. ``apply-translation`` --- also translate the body to its lab-frame
   position (defaults to ``false``):

    *boolean*
