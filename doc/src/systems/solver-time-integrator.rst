************************
[solver-time-integrator]
************************

Parameterises the time-integration scheme used by the solver.

Common Options
==============

The following options are common to both explicit and implicit
formulations:

#. ``formulation`` --- time-stepping formulation

    ``explicit`` | ``implicit``

    Defaults to ``explicit`` if not specified.

#. ``tstart`` --- initial time

    *float*

#. ``tend`` --- final time

    *float*

#. ``dt`` --- time-step

    *float*

#. ``dt-min`` --- minimum permissible time-step

    *float* (default: 1e-12)

#. ``dt-lookahead`` --- number of steps over which to smooth the
    time-step when approaching a target time (e.g., a plugin output
    time). Distributes the remaining time into equal steps to avoid
    a single undersized step at the end.

    *int* (default: 10)

Explicit Formulation
====================

For explicit time-stepping (``formulation = explicit`` or omitted):

#. ``scheme`` --- time-integration scheme

    ``euler`` | ``rk34`` | ``rk4`` | ``rk45`` | ``tvd-rk3``

#. ``controller`` --- time-step controller

    ``none`` | ``pi`` | ``cfl``

    where ``pi`` only works with ``rk34`` and ``rk45`` and requires

    - ``atol`` --- absolute error tolerance

        *float*

    - ``atol-<var>`` --- per-variable absolute tolerance

        *float*

        For problems with disparate variable scales (e.g., high-pressure
        compressible flow where energy is much larger than density),
        per-variable tolerances can be used instead of a scalar ``atol``.
        If any ``atol-<var>`` is set, all must be set. Variable names
        depend on the system; for Euler and Navier-Stokes in 2D these
        are ``rho``, ``rhou``, ``rhov``, ``E``.

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

    - ``auto-dt-max`` --- attempt to learn ``dt-max`` from rejected
        time step information

        ``true`` | ``false`` (default: ``false``)

    and ``cfl`` works with all explicit schemes and computes the
    time-step from the CFL condition as
    ``dt = cfl / (max_wavespeed * (2*order + 1))``. It requires

    - ``cfl`` --- CFL number

        *float*

        Maximum stable values for linear advection are:

        ========== =====
        Scheme     CFL
        ========== =====
        euler      1.00
        tvd-rk3    2.51
        rk4        2.79
        rk34       2.79
        rk45       4.82
        ========== =====

    - ``dt-max`` --- maximum permissible time-step

        *float* (default: 100.0)

    - ``cfl-nsteps`` --- recompute the CFL-based time-step every
        N accepted steps

        *int* (default: 1)

Example:

.. code-block:: ini

    [solver-time-integrator]
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

Per-variable tolerances for problems with large dynamic range:

.. code-block:: ini

    [solver-time-integrator]
    scheme = rk45
    controller = pi
    tstart = 0.0
    tend = 10.0
    dt = 0.001
    atol-rho = 1e-8
    atol-rhou = 1e-6
    atol-rhov = 1e-8
    atol-E = 1e-3
    rtol = 0.00001
    errest-norm = l2
    safety-fact = 0.9
    min-fact = 0.3
    max-fact = 2.5

CFL-based time-stepping:

.. code-block:: ini

    [solver-time-integrator]
    scheme = rk45
    controller = cfl
    tstart = 0.0
    tend = 10.0
    dt = 0.001
    cfl = 2.4

Implicit Formulation
====================

For implicit time-stepping (``formulation = implicit``), PyFR employs
a Jacobian-free Newton-Krylov (JFNK) method. Each time step requires
solving a nonlinear system using Newton's method, where the linear
systems arising at each Newton iteration are solved using a Krylov
subspace method.  Linearly implicit (Rosenbrock-W) schemes instead
linearise about the start of the step and solve one linear system per
stage with the same Krylov machinery, bypassing the nonlinear solver.

Scheme Selection
----------------

#. ``scheme`` --- implicit time-integration scheme

    ``euler`` | ``trapezium`` | ``trbdf2`` | ``kvaerno43`` | ``ros2`` |
    ``ros34pw2``

    where

    - ``euler`` --- first-order backward Euler (L-stable)
    - ``trapezium`` --- second-order trapezoidal rule (A-stable)
    - ``trbdf2`` --- second-order TR-BDF2 with embedded first-order
      error estimator (L-stable)
    - ``kvaerno43`` --- third-order ESDIRK with embedded second-order
      error estimator (L-stable)
    - ``ros2`` --- second-order two-stage linearly implicit
      Rosenbrock-W scheme with embedded first-order error estimator
      (L-stable)
    - ``ros34pw2`` --- third-order four-stage linearly implicit
      Rosenbrock-W scheme with embedded second-order error estimator
      (L-stable)

    Schemes with embedded error estimators (i.e., ``trbdf2``,
    ``kvaerno43``, ``ros2`` and ``ros34pw2``) are required for
    adaptive time-stepping with the ``pi`` controller.

Controller Selection
--------------------

#. ``controller`` --- time-step controller

    ``none`` | ``pi`` | ``throughput``

    where ``pi`` requires a scheme with an embedded error estimator
    and the following additional parameters:

    - ``atol`` --- absolute error tolerance for step acceptance

        *float*

    - ``atol-<var>`` --- per-variable absolute tolerance

        *float*

        For problems with disparate variable scales, per-variable
        tolerances can be used instead of a scalar ``atol``. If any
        ``atol-<var>`` is set, all must be set. Variable names for
        Euler and Navier-Stokes: ``rho``, ``rhou``, ``rhov``, ``E``.

    - ``rtol`` --- relative error tolerance for step acceptance

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

        *float* (default: 100.0)

    - ``pi-alpha`` --- PI controller alpha parameter

        *float* (default: 0.7)

    - ``pi-beta`` --- PI controller beta parameter

        *float* (default: 0.4)

    and ``throughput`` adaptively adjusts the time step to maximise
    wall-clock throughput using a Gaussian process (GP) optimiser.
    It does not require an embedded error estimator. The GP search
    range is bounded by ``dt-min`` and ``dt-max``. Additional
    parameters:

    - ``dt-max`` --- maximum permissible time-step

        *float* (default: 100.0)

    - ``dt-update-interval`` --- number of steps per evaluation
        window; the GP is updated once per window

        *int* (default: 100)

    - ``growth-fact`` --- growth factor used during the initial
        exploration phase

        *float* (default: 1.2)

    - ``dt-reexplore-interval`` --- number of windows over which a
        throughput observation loses its weight in the model; a held
        time step re-measures a neighbouring time step every half interval

        *int* (default: 50)

    After each window the measured throughput, and its sampling
    variance, are added to the model and the time step is moved one
    growth step towards the optimum.  While the optimum lies at the
    edge of the data the time step is moved past it, so that the
    model keeps exploring until the throughput stops improving.
    Windows needing no more than one nonlinear iteration per stage are
    not recorded and the time step is simply grown, and steps taken
    while a ``linear-tol-controller`` is probing are excluded.  A
    window, or a failed step, whose throughput falls outside the
    confidence band of the model's prediction marks a change in the
    landscape, and the model is started afresh from it.

    A step on which the nonlinear solver fails to converge is
    abandoned.  If the preconditioner predates the step it is rebuilt
    and the step retried; otherwise ``pi``, ``cfl``, and
    ``throughput`` reduce the time step and retry while ``none``
    raises an error.  The response is governed by:

    - ``failure-fact`` --- time-step reduction factor on nonlinear
        solver failure

        *float* (default: 0.5)

    - ``max-failures`` --- maximum consecutive nonlinear solver
        failures before raising an error

        *int* (default: 5)

Linear (Krylov) Solver Options
------------------------------

These apply only when ``nonlinear-solver = newton``.

#. ``linear-solver`` --- Krylov subspace method

    ``gmres`` | ``bicgstab`` | ``tfqmr`` (default: ``gmres``)

#. ``linear-max-iter`` --- maximum matrix-vector products per solve

    *int* (default: 10)

#. ``linear-rtol`` --- relative tolerance for Krylov convergence

    *float* (default: 1e-2)

#. ``linear-tol-controller`` --- Krylov tolerance controller

    ``none`` | ``windowed-gp`` | ``list`` (default: ``none``)

    When set to ``windowed-gp``, the Krylov tolerance is
    adaptively adjusted using a GP-based optimiser to balance
    solve cost against accuracy, periodically re-probing the
    tolerance measured longest ago.  When set to ``list``, the
    controller cycles through a fixed set of candidate
    tolerances (see ``linear-rtol-list``) and exploits the
    one with lowest cost.  Both start from ``linear-rtol`` and
    cost each candidate against the incumbent over alternating
    time steps.  Newton never solves more tightly than it needs
    to converge, so the tolerance only matters on stages taking
    more than one iteration.

#. ``linear-probe-nsolves`` --- number of solves per probe
   window (used by ``windowed-gp`` and ``list``) when no throughput
   limiter is active; otherwise half a ``dt-update-interval`` window

    *int* (default: 20)

#. ``linear-rtol-min`` --- minimum Krylov tolerance for
   ``windowed-gp`` controller

    *float* (default: 1e-3)

#. ``linear-rtol-max`` --- maximum Krylov tolerance for
   ``windowed-gp`` controller

    *float* (default: 1e-1)

#. ``linear-max-retries`` --- maximum retries with a tightened
   tolerance on Newton failure (``windowed-gp`` controller)

    *int* (default: 2)

    A stage which stalls short of the tolerance is continued from its
    current iterate with the Krylov tolerance tightened by
    ``linear-retry-fact``; if it stalls again the step is abandoned.  A
    stage which diverges is restarted from its initial guess at the tighter
    tolerance, up to this many times.  No retry is made if every
    Krylov solve of the attempt used its full ``linear-max-iter``
    budget, as the tolerance was then not the limiting factor.  The
    tolerance controller is informed once per stage with the total
    time over all attempts.

#. ``linear-retry-fact`` --- factor applied to the Krylov tolerance
   on each retry

    *float* (default: 0.1)

#. ``linear-rtol-list`` --- candidate tolerances for ``list``
   controller

    *list of floats*

    Example: ``[1e-1, 3e-2, 1e-2, 3e-3, 1e-3]``.  Each
    candidate is probed for ``linear-probe-nsolves`` steps and
    the cheapest is used until the next re-probe cycle.

#. ``linear-reprobe-interval`` --- re-probe interval for
   ``list`` controller

    *int* (default: 200)

    Number of steps between re-probe cycles.

GMRES Options
-------------

Read from the ``[solver-gmres]`` section when ``linear-solver = gmres``:

#. ``arnoldi`` --- Arnoldi orthogonalisation method

    ``cgs`` | ``mgs`` (default: ``cgs``)

    Classical Gram-Schmidt (``cgs``) or Modified Gram-Schmidt
    (``mgs``).

#. ``restart`` --- GMRES restart size

    *int* (default: 0)

    When set to a positive integer *m*, GMRES restarts every *m*
    iterations.  Total iterations are still bounded by
    ``linear-max-iter``.  When 0 (default), no restart is
    performed.

Preconditioner Options
----------------------

#. ``precond`` --- preconditioner type

    ``none`` | ``block-jacobi`` (default: ``none``)

    where ``block-jacobi`` is an element-wise block Jacobi
    preconditioner.  The preconditioner is shared by the Newton
    Krylov solve and the Anderson fixed point.  It is built on first
    use and rebuilt after a nonlinear solver failure, once the wall
    time spent on Krylov work in excess of that of the first window of
    steps after a build exceeds the build time, or as soon as
    :math:`\gamma \Delta t` differs by more than half a decade from
    its value at the build.

#. ``precond-check-interval`` --- number of accepted steps at the
   incumbent Krylov tolerance which set the baseline for preconditioner
   drift

    *int* (default: 50)

#. ``precond-precision`` --- preconditioner storage precision

    ``double`` | ``single`` | ``half`` | ``bf16`` (default: backend
    precision)

    Controls the precision used for storing and applying the
    block-Jacobi preconditioner.  Lower precision reduces memory
    and can improve throughput at the cost of preconditioner quality.
    Automatically clamped to the backend precision if set higher.

    The finite-difference Jacobian is always evaluated at the backend
    precision.  For each setting the intermediate Jacobian storage, the
    precision the block inverse is computed in, and the inverse storage
    are:

    ``double``
        Jacobian and inverse stored in double; inverse computed in
        double.  Clamped to ``single`` on a single-precision backend.

    ``single``
        Jacobian and inverse stored in single; inverse computed in
        single.

    ``half``
        Jacobian stored in single; inverse computed in single and
        stored in half.

    ``bf16``
        Jacobian stored in bf16; inverse computed in single and stored
        in bf16.

#. ``fd-eps`` --- finite-difference perturbation

    ``auto`` or *float* (default: ``auto``)

    Finite-difference perturbation shared by the JFNK matrix-vector
    product and the block-Jacobi construction.  ``auto`` selects an
    adaptive value from the solution norm; a numeric value pins it.

#. ``solve-atol`` --- absolute tolerance for the stage solves

    *float* (default: 1e-8)

    Sets the floor of the nonlinear convergence tolerance and the
    scaling of the residual seen by the Krylov solver.

#. ``solve-atol-<var>`` --- per-variable absolute tolerance

    *float*

    For problems with disparate variable scales (e.g., high-pressure
    compressible flow where energy is much larger than density),
    per-variable tolerances can improve convergence. The variable
    names depend on the system; for Euler and Navier-Stokes in 2D
    these are ``rho``, ``rhou``, ``rhov``, ``E``. Example:

    .. code-block:: ini

        solve-atol-rho = 1e-8
        solve-atol-rhou = 1e-6
        solve-atol-rhov = 1e-8
        solve-atol-E = 1e-3

Nonlinear Solver Options
------------------------

#. ``nonlinear-solver`` --- nonlinear solver driving each stage

    ``newton`` | ``anderson`` (default: ``newton``)

    ``newton`` is a Jacobian-free Newton-Krylov (JFNK) solver that
    drives an inner linear (Krylov) solve; ``anderson`` is a
    preconditioned Anderson acceleration of the defect-correction
    fixed point and uses no inner Krylov solver.

#. ``nonlinear-rtol`` --- relative tolerance for nonlinear convergence

    *float* (default: 1e-4)

#. ``nonlinear-max-iter`` --- maximum nonlinear iterations per stage

    *int* (default: 10)

#. ``nonlinear-stall-fact`` --- factor above the nonlinear tolerance
   within which a stalled residual is accepted as converged

    *float* (default: 2.0)

    The solve stops once the rate of convergence over the last
    ``nonlinear-stall-iters`` iterations can not reach the tolerance
    within the remaining iterations, as happens at the
    finite-difference noise floor.  A stalled residual within this
    factor of the tolerance is deemed converged; otherwise the step
    is abandoned as described under ``controller``.

#. ``nonlinear-stall-iters`` --- number of iterations over which
   the rate of convergence is measured when detecting a stall

    *int* (default: 3)

#. ``nonlinear-diverge-fact`` --- growth of the residual relative to
   the initial guess at which a solve is deemed to have diverged

    *float* (default: 10.0)

#. ``nonlinear-linesearch`` --- enable backtracking line search

    ``true`` | ``false`` (default: ``false``)

    A backtracking line search can improve robustness for difficult
    problems by ensuring the step reduces the residual.  It applies
    to both the Newton and Anderson solvers, adds computational
    overhead, and is not necessary for well-conditioned problems.

#. ``nonlinear-linesearch-max-iter`` --- maximum line search iterations

    *int* (default: 5)

#. ``nonlinear-linesearch-fact`` --- line search reduction factor

    *float* (default: 0.5)

#. ``nonlinear-linesearch-c1`` --- Armijo condition constant

    *float* (default: 1e-4)

Anderson Options
----------------

Read from the ``[solver-anderson]`` section when
``nonlinear-solver = anderson``:

#. ``depth`` --- Anderson history depth *m*

    *int* (default: 5)

    The number of previous iterates retained for the acceleration
    least-squares.  A depth of 0 reduces to preconditioned Picard
    iteration.

Example (Fixed Time-Step)
-------------------------

A basic implicit configuration with fixed time-step:

.. code-block:: ini

    [solver-time-integrator]
    formulation = implicit
    scheme = trapezium
    controller = none
    tstart = 0.0
    tend = 10.0
    dt = 0.005

    linear-solver = gmres
    linear-max-iter = 50
    linear-rtol = 1e-3
    precond = none

Example (Adaptive Time-Step --- PI)
------------------------------------

An implicit configuration with adaptive time-stepping using
the PI controller:

.. code-block:: ini

    [solver-time-integrator]
    formulation = implicit
    scheme = kvaerno43
    controller = pi
    tstart = 0.0
    tend = 10.0
    dt = 0.01
    dt-max = 0.1
    atol = 1e-5
    rtol = 1e-5

    linear-solver = gmres
    linear-max-iter = 50
    linear-rtol = 1e-3
    precond = none

Example (Adaptive Time-Step --- Throughput)
--------------------------------------------

An implicit configuration with throughput-based adaptive
time-stepping:

.. code-block:: ini

    [solver-time-integrator]
    formulation = implicit
    scheme = trapezium
    controller = throughput
    tstart = 0.0
    tend = 10.0
    dt = 0.01
    dt-max = 1.0

    linear-solver = gmres
    linear-max-iter = 50
    linear-rtol = 1e-3
    precond = none

Example (Per-Variable Scaling)
------------------------------

For high-pressure compressible flow with large dynamic range:

.. code-block:: ini

    [solver-time-integrator]
    formulation = implicit
    scheme = trapezium
    controller = none
    tstart = 0.0
    tend = 1.0
    dt = 0.0001

    linear-solver = gmres
    linear-max-iter = 50
    linear-rtol = 1e-3
    precond = none

    nonlinear-rtol = 1e-4
    solve-atol-rho = 1e-8
    solve-atol-rhou = 1e-6
    solve-atol-rhov = 1e-8
    solve-atol-E = 1e-3
