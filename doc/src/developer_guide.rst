***************
Developer Guide
***************

======================================
A Brief Overview of the PyFR Framework
======================================

Where to Start
--------------

The symbolic link :code:`pyfr.scripts.pyfr` points to the script
:code:`pyfr.scripts.main`, which is where it all starts! Specifically,
the function :code:`process_run` calls the function
:code:`_process_common`, which in turn calls the function
:code:`get_solver`, returning an Integrator -- a composite of a
`Controller`_ and a `Stepper`_. The Integrator has a method named
:code:`run`, which is then called to run the simulation.

Controller
----------

A `Controller`_ acts to advance the simulation in time. Specifically, a
`Controller`_ has a method named :code:`advance_to` which advances a
`System`_ to a specified time. There are two types of `Controller`_
available in PyFR |release|:

.. toggle-header::
    :header: *StdNoneController* **Click to show**

    .. autoclass:: pyfr.integrators.std.controllers.StdNoneController
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:

.. toggle-header::
    :header: *StdPIController* **Click to show**

    .. autoclass:: pyfr.integrators.std.controllers.StdPIController
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:

|

Types of `Controller`_ are related via the following inheritance diagram:

.. inheritance-diagram:: pyfr.integrators.std.controllers
    :parts: 1

|

Stepper
-------

A `Stepper`_ acts to advance the simulation by a single time-step.
Specifically, a `Stepper`_ has a method named :code:`step` which
advances a `System`_ by a single time-step. There are five types of
`Stepper`_ available in PyFR |release|:

.. toggle-header::
    :header: *StdEulerStepper* **Click to show**

    .. autoclass:: pyfr.integrators.std.steppers.StdEulerStepper
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:

.. toggle-header::
    :header: *StdRK4Stepper* **Click to show**

    .. autoclass:: pyfr.integrators.std.steppers.StdRK4Stepper
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:

.. toggle-header::
    :header: *StdRK34Stepper* **Click to show**

    .. autoclass:: pyfr.integrators.std.steppers.StdRK34Stepper
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:

.. toggle-header::
    :header: *StdRK45Stepper* **Click to show**

    .. autoclass:: pyfr.integrators.std.steppers.StdRK45Stepper
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:

.. toggle-header::
    :header: *StdTVDRK3Stepper* **Click to show**

    .. autoclass:: pyfr.integrators.std.steppers.StdTVDRK3Stepper
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:

|

Types of `Stepper`_ are related via the following inheritance diagram:

.. inheritance-diagram:: pyfr.integrators.std.steppers
    :parts: 1

|

System
------

A `System`_ holds information/data for the system, including
`Elements`_, `Interfaces`_, and the `Backend`_ with which the
simulation is to run. A `System`_ has a method named :code:`rhs`, which
obtains the divergence of the flux (the 'right-hand-side') at each
solution point. The method :code:`rhs` invokes various kernels which
have been pre-generated and loaded into queues. A `System`_ also has a
method named :code:`_gen_kernels` which acts to generate all the
kernels required by a particular `System`_. A kernel is an instance of
a 'one-off' class with a method named :code:`run` that implements the
required kernel functionality. Individual kernels are produced by a
kernel provider. PyFR |release| has various types of kernel provider. A
`Pointwise Kernel Provider`_ produces point-wise kernels such as
Riemann solvers and flux functions etc. These point-wise kernels are
specified using an in-built platform-independent templating language
derived from `Mako <http://www.makotemplates.org/>`_, henceforth
referred to as `PyFR-Mako`_. There are two types of `System`_ available
in PyFR |release|:

.. toggle-header::
    :header: *EulerSystem* **Click to show**

    .. autoclass:: pyfr.solvers.euler.system.EulerSystem
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:

.. toggle-header::
    :header: *NavierStokesSystem* **Click to show**

    .. autoclass:: pyfr.solvers.navstokes.system.NavierStokesSystem
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:

|

Types of `System`_ are related via the following inheritance diagram:

.. inheritance-diagram:: pyfr.solvers.navstokes.system
                         pyfr.solvers.euler.system
    :top-classes: pyfr.solvers.base.system.BaseSystem
    :parts: 1

|

Elements
--------

An `Elements`_ holds information/data for a group of elements. There are
two types of `Elements`_ available in PyFR |release|:

.. toggle-header::
    :header: *EulerElements* **Click to show**

    .. autoclass:: pyfr.solvers.euler.elements.EulerElements
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:

.. toggle-header::
    :header: *NavierStokesElements* **Click to show**

    .. autoclass:: pyfr.solvers.navstokes.elements.NavierStokesElements
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:

|

Types of `Elements`_ are related via the following inheritance diagram:

.. inheritance-diagram:: pyfr.solvers.navstokes.elements
                         pyfr.solvers.euler.elements
    :top-classes: pyfr.solvers.base.elements.BaseElements
    :parts: 1

|

Interfaces
----------

An `Interfaces`_ holds information/data for a group of interfaces. There
are four types of (non-boundary) `Interfaces`_ available in PyFR
|release|:

.. toggle-header::
    :header: *EulerIntInters* **Click to show**

    .. autoclass:: pyfr.solvers.euler.inters.EulerIntInters
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:

.. toggle-header::
    :header: *EulerMPIInters* **Click to show**

    .. autoclass:: pyfr.solvers.euler.inters.EulerMPIInters
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:

.. toggle-header::
    :header: *NavierStokesIntInters* **Click to show**

    .. autoclass:: pyfr.solvers.navstokes.inters.NavierStokesIntInters
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:

.. toggle-header::
    :header: *NavierStokesMPIInters* **Click to show**

    .. autoclass:: pyfr.solvers.navstokes.inters.NavierStokesMPIInters
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:

|

Types of (non-boundary) `Interfaces`_ are related via the following
inheritance diagram:

.. inheritance-diagram:: pyfr.solvers.navstokes.inters.NavierStokesMPIInters
                         pyfr.solvers.navstokes.inters.NavierStokesIntInters
                         pyfr.solvers.euler.inters.EulerMPIInters
                         pyfr.solvers.euler.inters.EulerIntInters
    :top-classes: pyfr.solvers.base.inters.BaseInters
    :parts: 1

|

Backend
-------

A `Backend`_ holds information/data for a backend. There are five types
of `Backend`_ available in PyFR |release|:

.. toggle-header::
    :header: *CUDABackend* **Click to show**

    .. autoclass:: pyfr.backends.cuda.base.CUDABackend
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:

.. toggle-header::
    :header: *HIPBackend* **Click to show**

    .. autoclass:: pyfr.backends.hip.base.HIPBackend
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:

.. toggle-header::
    :header: *OpenCLBackend* **Click to show**

    .. autoclass:: pyfr.backends.opencl.base.OpenCLBackend
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:

.. toggle-header::
    :header: *OpenMPBackend* **Click to show**

    .. autoclass:: pyfr.backends.openmp.base.OpenMPBackend
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:

.. toggle-header::
    :header: *MetalBackend* **Click to show**

    .. autoclass:: pyfr.backends.metal.base.MetalBackend
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:

|

Types of `Backend`_ are related via the following inheritance diagram:


.. inheritance-diagram:: pyfr.backends.cuda.base
                         pyfr.backends.hip.base
                         pyfr.backends.opencl.base
                         pyfr.backends.openmp.base
                         pyfr.backends.metal.base
    :top-classes: pyfr.backends.base.base.BaseBackend
    :parts: 1

|

Pointwise Kernel Provider
-------------------------

A `Pointwise Kernel Provider`_ produces point-wise kernels.
Specifically, a `Pointwise Kernel Provider`_ has a method named
:code:`register`, which adds a new method to an instance of a
`Pointwise Kernel Provider`_. This new method, when called, returns a
kernel. A kernel is an instance of a 'one-off' class with a method
named :code:`run` that implements the required kernel functionality.
The kernel functionality itself is specified using `PyFR-Mako`_. Hence,
a `Pointwise Kernel Provider`_ also has a method named
:code:`_render_kernel`, which renders `PyFR-Mako`_ into low-level
platform-specific code. The :code:`_render_kernel` method first sets
the context for Mako (i.e. details about the `Backend`_ etc.) and then
uses Mako to begin rendering the `PyFR-Mako`_ specification. When Mako
encounters a :code:`pyfr:kernel` an instance of a `Kernel Generator`_
is created, which is used to render the body of the
:code:`pyfr:kernel`. There are four types of `Pointwise Kernel
Provider`_ available in PyFR |release|:

.. toggle-header::
    :header: *CUDAPointwiseKernelProvider* **Click to show**

    .. autoclass:: pyfr.backends.cuda.provider.CUDAPointwiseKernelProvider
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:

.. toggle-header::
    :header: *HIPPointwiseKernelProvider* **Click to show**

    .. autoclass:: pyfr.backends.hip.provider.HIPPointwiseKernelProvider
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:

.. toggle-header::
    :header: *OpenCLPointwiseKernelProvider* **Click to show**

    .. autoclass:: pyfr.backends.opencl.provider.OpenCLPointwiseKernelProvider
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:

.. toggle-header::
    :header: *OpenMPPointwiseKernelProvider* **Click to show**

    .. autoclass:: pyfr.backends.openmp.provider.OpenMPPointwiseKernelProvider
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:

.. toggle-header::
    :header: *MetalPointwiseKernelProvider* **Click to show**

    .. autoclass:: pyfr.backends.metal.provider.MetalPointwiseKernelProvider
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:

|

Types of `Pointwise Kernel Provider`_ are related via the following
inheritance diagram:

.. inheritance-diagram:: pyfr.backends.openmp.provider
                         pyfr.backends.cuda.provider
                         pyfr.backends.hip.provider
                         pyfr.backends.opencl.provider
                         pyfr.backends.metal.provider
                         pyfr.backends.base.kernels.BasePointwiseKernelProvider
    :top-classes: pyfr.backends.base.kernels.BaseKernelProvider
    :parts: 1

|

Kernel Generator
----------------

A `Kernel Generator`_ renders the `PyFR-Mako`_ in a :code:`pyfr:kernel`
into low-level platform-specific code. Specifically, a `Kernel
Generator`_ has a method named :code:`render`, which applies `Backend`_
specific regex and adds `Backend`_ specific 'boiler plate' code to
produce the low-level platform-specific source -- which is compiled,
linked, and loaded. There are four types of `Kernel Generator`_
available in PyFR |release|:

.. toggle-header::
    :header: *CUDAKernelGenerator* **Click to show**

    .. autoclass:: pyfr.backends.cuda.generator.CUDAKernelGenerator
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:

.. toggle-header::
    :header: *HIPKernelGenerator* **Click to show**

    .. autoclass:: pyfr.backends.hip.generator.HIPKernelGenerator
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:

.. toggle-header::
    :header: *OpenCLKernelGenerator* **Click to show**

    .. autoclass:: pyfr.backends.opencl.generator.OpenCLKernelGenerator
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:

.. toggle-header::
    :header: *OpenMPKernelGenerator* **Click to show**

    .. autoclass:: pyfr.backends.openmp.generator.OpenMPKernelGenerator
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:

.. toggle-header::
    :header: *MetalKernelGenerator* **Click to show**

    .. autoclass:: pyfr.backends.metal.generator.MetalKernelGenerator
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:

|

Types of `Kernel Generator`_ are related via the following inheritance diagram:

.. inheritance-diagram:: pyfr.backends.cuda.generator.CUDAKernelGenerator
                         pyfr.backends.opencl.generator.OpenCLKernelGenerator
                         pyfr.backends.openmp.generator.OpenMPKernelGenerator
                         pyfr.backends.hip.generator.HIPKernelGenerator
                         pyfr.backends.metal.generator.MetalKernelGenerator
    :top-classes: pyfr.backends.base.generator.BaseKernelGenerator
    :parts: 1

|

=========
PyFR-Mako
=========

PyFR-Mako Kernels
-----------------

PyFR-Mako kernels are specifications of point-wise functionality that
can be invoked directly from within PyFR. They are opened with a header
of the form:

.. code-block:: none

    <%pyfr:kernel name='kernel-name' ndim='data-dimensionality' [argument-name='argument-intent argument-attribute argument-data-type' ...]>

where

1. ``kernel-name`` --- name of kernel

    *string*

2. ``data-dimensionality`` --- dimensionality of data

    *int*

3. ``argument-name`` --- name of argument

    *string*

4. ``argument-intent`` --- intent of argument

    ``in`` | ``out`` | ``inout``

5. ``argument-attribute`` --- attribute of argument

    ``mpi`` | ``scalar`` | ``view``

6. ``argument-data-type`` --- data type of argument

    *string*

and are closed with a footer of the form:

.. code-block:: none

    </%pyfr:kernel>

PyFR-Mako Macros
----------------

PyFR-Mako macros are specifications of point-wise functionality that
cannot be invoked directly from within PyFR, but can be embedded into
PyFR-Mako kernels. PyFR-Mako macros can be viewed as building blocks
for PyFR-mako kernels. They are opened with a header of the form:

.. code-block:: none

    <%pyfr:macro name='macro-name' params='param1, param2, ..., py:arg1, ...'>

where

1. ``macro-name`` --- name of macro

    *string*

2. ``param1, param2, ..., py:arg1, ...`` --- macro parameter/argument names

    *string*

and are closed with a footer of the form:

.. code-block:: none

    </%pyfr:macro>

Macro params can be either regular parameters (source code variables) or
Python arguments (when prefixed with ``py:``). Python arguments receive Python
objects that can be accessed during template rendering. All parameters prefixed
with ``py:`` must be accessed in the macro body within the usual ``${}``
expression *without the py: prefix*.

PyFR-Mako macros are embedded within a kernel using an expression of
the following form:

.. code-block:: none

    ${pyfr.expand('macro-name', 'value1', ..., data1, ..., param2='value2', ..., arg1=value, ...)};

where

1. ``macro-name`` --- name of the macro

    *string*

2. ``'value1', ...`` --- positional values for regular parameters

    *string*

3. ``data1, ...`` --- positional Python data

    *Python object*

4. ``param2='value2', ...`` --- keyword arguments for regular parameter

    *string* or *compilable*

5. ``arg1=value, ...`` --- keyword arguments for Python arguments

    *Python object*

Syntax
------

Basic Functionality
^^^^^^^^^^^^^^^^^^^

Basic functionality can be expressed using a restricted subset of the C
programming language. Specifically, use of the following is allowed:

1. ``+,-,*,/`` --- basic arithmetic

2. ``sin, cos, tan`` --- basic trigonometric functions

3. ``exp`` --- exponential

4. ``pow`` --- power

5. ``fabs`` --- absolute value

6. ``output = ( condition ? satisfied : unsatisfied )`` --- ternary if

7. ``min`` --- minimum

8. ``max`` --- maximum

However, conditional if statements, as well as for/while loops, are
not allowed.

Expression Substitution
^^^^^^^^^^^^^^^^^^^^^^^

Mako expression substitution can be used to facilitate PyFR-Mako kernel
specification. A Python expression :code:`expression` prescribed thus
:code:`${expression}` is substituted for the result when the PyFR-Mako
kernel specification is interpreted at runtime.

Example:

.. code-block:: none

    E = s[${ndims - 1}]

Conditionals
^^^^^^^^^^^^

Mako conditionals can be used to facilitate PyFR-Mako kernel
specification. Conditionals are opened with :code:`% if condition:` and
closed with :code:`% endif`. Note that such conditionals are evaluated
when the PyFR-Mako kernel specification is interpreted at runtime, they
are not embedded into the low-level kernel.

Example:

.. code-block:: none

    % if ndims == 2:
        fout[0][1] += t_xx;     fout[1][1] += t_xy;
        fout[0][2] += t_xy;     fout[1][2] += t_yy;
        fout[0][3] += u*t_xx + v*t_xy + ${-c['mu']*c['gamma']/c['Pr']}*T_x;
        fout[1][3] += u*t_xy + v*t_yy + ${-c['mu']*c['gamma']/c['Pr']}*T_y;
    % endif

Loops
^^^^^

Mako loops can be used to facilitate PyFR-Mako kernel specification.
Loops are opened with :code:`% for condition:` and closed with :code:`%
endfor`. Note that such loops are unrolled when the PyFR-Mako kernel
specification is interpreted at runtime, they are not embedded into the
low-level kernel.

Example:

.. code-block:: none

    % for i in range(ndims):
        rhov[${i}] = s[${i + 1}];
        v[${i}] = invrho*rhov[${i}];
    % endfor
