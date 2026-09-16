********************
[solver-plugin-nirf]
********************

Solves the flow in a non-inertial reference frame attached to a moving
body.  The mesh remains fixed in the body frame and the fictitious
body forces arising from the frame's translational acceleration,
rotation (centrifugal and Coriolis) and angular acceleration (Euler)
are added as momentum source terms, with the corresponding work term
added to the energy equation.  Boundary and initial conditions are
specified in the inertial (lab) frame as usual and are transformed into
the body frame automatically; see :doc:`postproc-plugins` for exporting
the solution back into the lab frame.

Two modes of operation are available, selected with ``motion``:

- ``prescribed`` --- the frame motion is given analytically as
  expressions in ``t``.
- ``free`` --- the frame motion is obtained by integrating the
  rigid-body equations of motion driven by the aerodynamic force and
  moment on a designated boundary.

Both modes serialise the kinematic state on checkpoint, so a
``prescribed`` run may be restarted in ``free`` mode, and vice versa,
without loss of continuity.

Common options
==============

#. ``motion`` --- mode of operation:

    ``prescribed`` | ``free``

#. ``center-of-rot`` --- centre of rotation, also used as the moment
   reference point (defaults to the origin):

    ``(x, y, [z])``

Prescribed motion
=================

All parameters are expressions in ``t``.  The translational velocity
and acceleration and the angular velocity and acceleration are derived
from them by symbolic differentiation.

#. ``frame-loc-{x,y,[z]}`` --- frame position:

    *string*

#. ``frame-rot-z`` --- rotation angle about *z* (2D):

    *string*

#. ``frame-rot-{x,y,z}`` --- ZYX Euler angles (3D):

    *string*

Free motion
===========

The body responds to the aerodynamic loads integrated over the
boundary named by ``boundary``.  The rigid-body equations are
integrated with Heun's method; in 3D the gyroscopic term is included.

#. ``mass`` --- body mass:

    *float*

#. ``inertia`` --- moment of inertia about *z* (2D) or the inertia
   tensor as a flat nine-element tuple (3D):

    *float* | ``(Ixx, Ixy, Ixz, Iyx, Iyy, Iyz, Izx, Izy, Izz)``

#. ``dof`` --- comma-separated active degrees of freedom, any subset
   of ``x, y, rz`` (2D) or ``x, y, z, rx, ry, rz`` (3D), defaulting
   to all:

    *string*

#. ``dt-ode`` --- interval at which the body is advanced.  The loads
   are integrated over ``boundary`` and the rigid-body equations
   stepped only once the solution has advanced by this much since the
   last update, defaults to every solver step:

    *float*

   .. note::

    Often, the ``dt-ode`` can be orders of magnitude larger than the solver time step. For best performance, increase ``dt-ode`` as much as possible while still resolving the body motion.

#. ``frame-loc0`` --- initial frame position:

    ``(x, y, [z])``

#. ``frame-velo0`` --- initial translational velocity:

    ``(u, v, [w])``

#. ``frame-accel0`` --- initial translational acceleration:

    ``(ax, ay, [az])``

#. ``frame-rot0-euler`` --- initial rotation angle (2D) or ZYX Euler
   angles (3D); mutually exclusive with ``frame-rot0-quat``:

    *float* | ``(phi, theta, psi)``

#. ``frame-rot0-quat`` --- initial orientation as a quaternion, which
   is normalised; mutually exclusive with ``frame-rot0-euler``:

    ``(w, x, y, z)``

#. ``frame-omega0`` --- initial angular velocity about *z* (2D) or
   vector (3D):

    *float* | ``(wx, wy, wz)``

#. ``frame-alpha0`` --- initial angular acceleration about *z* (2D)
   or vector (3D):

    *float* | ``(ax, ay, az)``

Force and trajectory output
===========================

In ``free`` mode the force integration is always active as it drives
the motion; in ``prescribed`` mode it is enabled by supplying both a
``boundary`` and an output ``file``.  The frame state and the boundary
rotation matrix are updated every solver step regardless of the
output cadence; only the file write is throttled.

#. ``boundary`` --- boundary over which to integrate the pressure and
   viscous stress; may be a brace enumeration such as ``{wall, fin}``,
   in which case the named boundaries are integrated as a single
   surface (required in ``free`` mode):

    *string*

#. ``nsteps-out`` --- write the CSV every ``nsteps-out`` solver steps
   (defaults to 1; mutually exclusive with ``dt-out``):

    *int*

#. ``dt-out`` --- write the CSV every ``dt-out`` seconds (mutually
   exclusive with ``nsteps-out``):

    *float*

#. ``file`` --- output CSV path:

    *string*

#. ``quad-deg`` --- degree of the surface quadrature (defaults to the
   solution order):

    *int*

.. note::

   For Navier--Stokes systems the viscous stress is integrated over
   every boundary named in ``boundary``, including slip walls on which
   it should vanish.  Name only no-slip walls when the viscous
   contribution matters.

Example:

.. code-block:: ini

    [solver-plugin-nirf]
    motion = free
    center-of-rot = (0.0, 0.0)
    mass = 1.0
    inertia = 0.1
    dof = y, rz
    boundary = {wall, fin}
    dt-ode = 1e-4
    file = trajectory.csv
    dt-out = 1e-3
