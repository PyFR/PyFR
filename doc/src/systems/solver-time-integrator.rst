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
subspace method.

Scheme Selection
----------------

#. ``scheme`` --- implicit time-integration scheme

    ``euler`` | ``trapezium`` | ``trbdf2`` | ``kvaerno43`` | ``esdirk32a``

    where

    - ``euler`` --- first-order backward Euler (L-stable)
    - ``trapezium`` --- second-order trapezoidal rule (A-stable)
    - ``trbdf2`` --- second-order TR-BDF2 (L-stable)
    - ``kvaerno43`` --- third-order ESDIRK with embedded second-order
      error estimator (L-stable)
    - ``esdirk32a`` --- third-order ESDIRK with embedded second-order
      error estimator (L-stable)

    Schemes with embedded error estimators (``kvaerno43``,
    ``esdirk32a``) are required for adaptive time-stepping with the
    ``pi`` controller.

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

    - ``failure-fact`` --- time-step reduction factor on Newton
        divergence

        *float* (default: 0.5)

    - ``max-failures`` --- maximum consecutive Newton failures
        before raising an error

        *int* (default: 5)

Krylov Solver Options
---------------------

#. ``krylov-solver`` --- Krylov subspace method

    ``gmres``

#. ``krylov-max-iter`` --- maximum Krylov iterations per solve

    *int* (default: 10)

#. ``krylov-rtol`` --- relative tolerance for Krylov convergence

    *float* (default: 1e-2)

#. ``krylov-precond`` --- preconditioner type

    ``none`` | ``block-jacobi``

    where ``block-jacobi`` is an element-wise block Jacobi
    preconditioner. For many problems, ``none`` is sufficient
    when using a modest ``krylov-rtol``.

#. ``krylov-tol-controller`` --- Krylov tolerance controller

    ``none`` | ``windowed-gp`` | ``list`` (default: ``none``)

    When set to ``windowed-gp``, the Krylov tolerance is
    adaptively adjusted using a GP-based optimiser to balance
    solve cost against accuracy.  When set to ``list``, the
    controller cycles through a fixed set of candidate
    tolerances (see ``krylov-rtol-list``) and exploits the
    one with lowest cost.

#. ``krylov-probe-nsolves`` --- number of solves per probe
   window (used by ``windowed-gp`` and ``list``)

    *int* (default: 20)

#. ``krylov-rtol-min`` --- minimum Krylov tolerance for
   ``windowed-gp`` controller

    *float* (default: 1e-3)

#. ``krylov-rtol-max`` --- maximum Krylov tolerance for
   ``windowed-gp`` controller

    *float* (default: 1e-1)

#. ``krylov-max-retries`` --- maximum retries with loosened
   tolerance on Newton failure (``windowed-gp`` controller)

    *int* (default: 2)

#. ``krylov-rtol-list`` --- candidate tolerances for ``list``
   controller

    *list of floats*

    Example: ``[1e-1, 3e-2, 1e-2, 3e-3, 1e-3]``.  Each
    candidate is probed for ``krylov-probe-nsolves`` steps and
    the cheapest is used until the next re-probe cycle.

#. ``krylov-reprobe-interval`` --- re-probe interval for
   ``list`` controller

    *int* (default: 200)

    Number of steps between re-probe cycles.

#. ``gmres-arnoldi`` --- Arnoldi orthogonalisation method

    ``cgs`` | ``mgs``

    Classical Gram-Schmidt (``cgs``) or Modified Gram-Schmidt
    (``mgs``). Default is ``cgs``.

#. ``gmres-restart`` --- GMRES restart size

    *int* (default: 0)

    When set to a positive integer *m*, GMRES restarts every *m*
    iterations.  Total iterations are still bounded by
    ``krylov-max-iter``.  When 0 (default), no restart is
    performed.

Newton Solver Options
---------------------

#. ``newton-rtol`` --- relative tolerance for Newton convergence

    *float* (default: 1e-4)

#. ``newton-atol`` --- absolute tolerance for Newton convergence

    *float* (default: 1e-8)

#. ``newton-max-iter`` --- maximum Newton iterations per stage

    *int* (default: 10)

#. ``newton-atol-<var>`` --- per-variable absolute tolerance

    *float*

    For problems with disparate variable scales (e.g., high-pressure
    compressible flow where energy is much larger than density),
    per-variable tolerances can improve convergence. The variable
    names depend on the system; for Euler and Navier-Stokes in 2D
    these are ``rho``, ``rhou``, ``rhov``, ``E``. Example:

    .. code-block:: ini

        newton-atol-rho = 1e-8
        newton-atol-rhou = 1e-6
        newton-atol-rhov = 1e-8
        newton-atol-E = 1e-3

Line Search
-----------

#. ``newton-linesearch`` --- enable backtracking line search

    ``true`` | ``false`` (default: ``false``)

    A backtracking line search can improve robustness for
    difficult problems by ensuring the Newton step reduces the
    residual. However, it adds computational overhead and is not
    necessary for well-conditioned problems.

#. ``newton-linesearch-max-iter`` --- maximum line search iterations

    *int* (default: 5)

#. ``newton-linesearch-fact`` --- line search reduction factor

    *float* (default: 0.5)

#. ``newton-linesearch-c1`` --- Armijo condition constant

    *float* (default: 1e-4)

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

    krylov-solver = gmres
    krylov-max-iter = 50
    krylov-rtol = 1e-3
    krylov-precond = none

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

    krylov-solver = gmres
    krylov-max-iter = 50
    krylov-rtol = 1e-3
    krylov-precond = none

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

    krylov-solver = gmres
    krylov-max-iter = 50
    krylov-rtol = 1e-3
    krylov-precond = none

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

    krylov-solver = gmres
    krylov-max-iter = 50
    krylov-rtol = 1e-3
    krylov-precond = none

    newton-rtol = 1e-4
    newton-atol-rho = 1e-8
    newton-atol-rhou = 1e-6
    newton-atol-rhov = 1e-8
    newton-atol-E = 1e-3
