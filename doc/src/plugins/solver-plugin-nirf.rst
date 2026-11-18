********************
[solver-plugin-nirf]
********************

Solves the flow in a non-inertial reference frame attached to a moving
body.  The mesh remains fixed in the body frame and the fictitious
body forces arising from the frame's translational acceleration,
rotation (centrifugal and Coriolis) and angular acceleration (Euler)
are added as momentum source terms, with the corresponding work term
added to the energy equation.  See :doc:`postproc-plugins` for
exporting the solution back into the lab frame.

Boundary and initial conditions are specified in the inertial (lab)
frame and transformed into the body frame automatically.  A boundary
condition which specifies a velocity together with a density or a
pressure describes the free stream and is taken to be given in the lab
frame: its velocity is a lab-frame velocity, from which the frame
motion is removed.  A boundary condition which specifies a velocity
alone, such as an isothermal wall, moves with the body and its velocity
is relative to the body.  Wherever the parameters of a boundary
condition depend on position, the positions are the lab-frame locations
of the boundary points.  A boundary condition which specifies its flow
direction by angles, ``sub-in-ftpttang``, is not transformed: its
direction and total conditions are those of the body frame.

Two modes of operation are available, selected with ``motion``:

- ``prescribed`` --- the frame motion is given analytically as
  expressions in ``t``.
- ``free`` --- the frame motion is obtained by integrating the
  rigid-body equations of motion driven by the aerodynamic force and
  moment on a designated boundary.

Both modes serialise the kinematic state on checkpoint, so a
``prescribed`` run may be restarted in ``free`` mode without loss of
continuity.  A ``free`` run restarted in ``prescribed`` mode takes its
state from the expressions, which must then match the stored state at
the restart time.

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
from them by symbolic differentiation.  The expressions may be
piecewise, through ``min`` and ``max``, provided their first
derivatives are continuous: the flow cannot follow a jump in velocity,
and a jump in acceleration reduces the time accuracy to first order at
that instant.  Where its arguments are equal the derivative of ``min``
or ``max`` is that of its first argument, so where two pieces meet
only one of them may name ``t`` first: ``min(t, t1)`` is joined by
``max(0, t - t1)``, not ``max(t - t1, 0)``.  Angles are in radians.

#. ``frame-loc-{x,y,[z]}`` --- frame position:

    *string*

#. ``frame-rot-z`` --- rotation angle about *z* (2D):

    *string*

#. ``frame-rot-{x,y,z}`` --- rotation angles about *x*, *y* and *z*,
   applied as the ZYX Euler sequence (3D):

    *string*

Free motion
===========

The body responds to the aerodynamic loads integrated over the
boundary named by ``boundary``.  The rigid-body equations are
integrated with Heun's method; in 3D the gyroscopic term is included.
The position, translational velocity and acceleration of the frame are
held in lab axes and its angular velocity and acceleration in body
axes, with ``center-of-rot`` taken to be the centre of mass.  A new
run takes its initial accelerations from the loads on the initial
solution.  A restart continues from the accelerations stored in the
solution file, which for a ``prescribed`` run are those of the
prescribed motion.

#. ``mass`` --- body mass:

    *float*

#. ``inertia`` --- moment of inertia about *z* (2D) or the inertia
   tensor as a flat nine-element tuple (3D):

    *float* | ``(Ixx, Ixy, Ixz, Iyx, Iyy, Iyz, Izx, Izy, Izz)``

#. ``dof`` --- comma-separated active degrees of freedom, any subset
   of ``x, y, rz`` (2D) or ``x, y, z, rx, ry, rz`` (3D), defaulting
   to all, or ``none`` for a body which does not respond to the loads.
   Translations are along the lab axes and rotations about the body
   axes:

    *string*

#. ``dt-ode`` --- interval at which the body is advanced.  The loads
   are integrated over ``boundary`` and the rigid-body equations
   stepped only once the solution has advanced by this much since the
   last update, defaults to every solver step:

    *float*

   .. note::

      Often, the ``dt-ode`` can be orders of magnitude larger than the
      solver time step.  For best performance, increase ``dt-ode`` as
      much as possible while still resolving the body motion.

#. ``frame-loc0`` --- initial frame position:

    ``(x, y, [z])``

#. ``frame-velo0`` --- initial translational velocity:

    ``(u, v, [w])``

#. ``frame-rot0-euler`` --- initial rotation angle (2D) or the
   rotation angles about *x*, *y* and *z* of the ZYX Euler sequence
   (3D), in radians; mutually exclusive with ``frame-rot0-quat``:

    *float* | ``(rx, ry, rz)``

#. ``frame-rot0-quat`` --- initial orientation as a quaternion, which
   is normalised and in 2D must be a rotation about *z*; mutually
   exclusive with ``frame-rot0-euler``:

    ``(w, x, y, z)``

#. ``frame-omega0`` --- initial angular velocity about *z* (2D) or
   vector (3D), in radians per unit time:

    *float* | ``(wx, wy, wz)``

Force and trajectory output
===========================

In ``free`` mode the force integration is always active as it drives
the motion; in ``prescribed`` mode it is enabled by supplying both a
``boundary`` and an output ``file``.  The frame state is advanced
independently of the output cadence; only the file write is
throttled.  Each record holds the time, the rotation angles about
*x*, *y* and *z* of the ZYX Euler sequence (about *z* alone in 2D),
the angular velocity and acceleration, the frame position, velocity
and acceleration, and the force and moment.  The force and moment are
reported in body axes, the moment being taken about ``center-of-rot``.

#. ``boundary`` --- boundary over which to integrate the pressure and
   viscous stress; may be a brace enumeration such as ``{wall, fin}``,
   in which case the named boundaries are integrated as a single
   surface (required in ``free`` mode):

    *string*

#. ``nsteps-out`` --- write to the file every ``nsteps-out`` solver
   steps (defaults to 1):

    *int*

#. ``dt-out`` --- write to the file every ``dt-out`` time units, in
   which case ``nsteps-out`` is ignored:

    *float*

#. ``file`` --- output file path; should the file already exist it
   will be appended to:

    *string*

#. ``file-format`` --- output file type (defaults to CSV):

    ``csv`` | ``hdf5``

#. ``header`` --- for CSV output if to write a header row or not:

    *boolean*

#. ``file-dataset`` --- for HDF5 output where in the HDF5 to write the
   data:

    *string*

#. ``file-reset`` --- if to clear any existing output when starting a
   new simulation; restarts always append:

    *boolean*

#. ``quad-deg`` --- degree of the surface quadrature (defaults to the
   solution order, and to one at order zero):

    *int*

#. ``quad-deg-{etype}`` --- degree of the surface quadrature for one
   interface type, in place of ``quad-deg``:

    *int*

#. ``quad-pts-{etype}`` --- name of quadrature rule (optional):

    *string*

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
