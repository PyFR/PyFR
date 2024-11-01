.. highlight:: none

**********
User Guide
**********

For information on how to install PyFR see :ref:`installation`.

.. _running-pyfr:

Running PyFR
============

PyFR |release| uses three distinct file formats:

#. .ini --- configuration file
#. .pyfrm --- mesh file
#. .pyfrs --- solution file

The following core commands are available from the ``pyfr`` program:

pyfr import
   Convert a `Gmsh <http:http://geuz.org/gmsh/>`_ .msh file into a PyFR
   .pyfrm file.  Example::

        pyfr import mesh.msh mesh.pyfrm

pyfr partition
   Handles mesh partitioning.

   pyfr partition add
      Adds a new partitioning to mesh.  Example::

         pyfr partition add mesh.pyfrm 10 ten_parts

      Here, ``ten_parts`` is the *name* of the partitioning and is an
      arbitrary identifier.  If no name is provided it defaults to the
      number of parts.

      For mixed grids one must include the ``-e`` flag followed by
      weights for each element type, or the ``balanced`` argument.
      Further details can be found in the
      :ref:`performance guide <perf mixed grids>`.

   pyfr partition reconstruct
      Reconstructs a partitioning from a solution file.  Example::

         pyfr partition reconstruct mesh.pyfrm soln.pyfrm part_name

   pyfr partition list
      Lists partitionings in a mesh.  Example::

         pyfr partition list mesh.pyfrm

   pyfr partition info
      Shows information about a specific partitioning in a mesh.
      Example::

         pyfr partition info mesh.pyfrm ten_parts

   pyfr partition remove
      Deletes a partitioning from a mesh.  Example::

         pyfr partition remove mesh.pyfrm ten_parts

pyfr run
   Start a new PyFR simulation.  Example::

        pyfr run mesh.pyfrm configuration.ini

pyfr restart
   Restart a PyFR simulation from an existing solution file.  Example::

        pyfr restart mesh.pyfrm solution.pyfrs

   It is also possible to restart with a different configuration file.
   Example::

        pyfr restart mesh.pyfrm solution.pyfrs configuration.ini

pyfr export
   Convert a PyFR ``.pyfrs`` file into an unstructured VTK ``.vtu`` or
   ``.pvtu`` file.
   
   pyfr export volume
      Exports a volume grid.  If ``--eopt=order:n`` is provided then
      PyFR elements are converted where possible to high-order VTK
      elements.  Example::

        pyfr export --eopt=order:4 mesh.pyfrm solution.pyfrs solution.vtu

      If a ``--eopt=divisor:n`` flag is provided with an integer
      argument then elements are subdivided into linear VTK cells.
      Example::

        pyfr export --eopt=divisor:4 mesh.pyfrm solution.pyfrs solution.vtu

      By default elements are converted to high-order VTK cells which
      are exported, where the order of the cells is equal to the order
      of the solution data in the file.

      Additionally, by default, all of the fields in the solution file
      will be exported. If only a specific field is desired this can be
      specified with the ``-f`` flag; for example ``-f density -f
      velocity`` will only export   the *density* and *velocity* fields.

   pyfr export boundary
      Exports one of more boundaries.  Example::

        pyfr export boundary mesh.pyfrm solution.pyfrms solution.vtu lower_wall upper_wall

      Note that boundary export is only supported for 3D grids.

   pyfr export stl
      Exports one or more STL surfaces.  Example::

        pyfr export boundary mesh.pyfrm solution.pyfrms solution.vtu teapot

      The STL surfaces must have already been added to the mesh with
      ``pyfr region add``.

   All of the export commands also support a *batch* processing mode
   wherein the list of input solution files and output files are read in
   from disk.  This option is activated by passing ``-`` for the
   solution file and output file.  By default, the file is taken to be
   *stdin* although this can be overridden with the ``--batchfile``
   option.  Example::

        for f in *.pyfrs; do echo "$f ${f%.pyfrs}.vtu"; done | pyfr -p export volume mesh.pyfrm - -

   This command will export each solution file in the current directory
   to a VTU file.

pyfr region
   Handles STL region processing.

   pyfr region add
      Adds an STL region to the mesh.  Example::

         pyfr partition add mesh.pyfrm teapot.stl teapot

   pyfr region list
      Lists the STL regions in the mesh.  Example::

         pyfr partition list mesh.pyfrm

   pyfr region remove
      Removes an STL region from the mesh.  Example::

         pyfr region remove mesh.pyfrm teapot

The ``run``, ``restart``, and ``export`` commands can be run in
parallel. To do so prefix ``pyfr`` with ``mpiexec -n <cores/devices>``.
Note that there must exist a partitioning in the mesh with an
appropriate number of parts.

.. _configuration-file:

Configuration File (.ini)
=========================

The .ini configuration file parameterises the simulation. It is written
in the `INI <http://en.wikipedia.org/wiki/INI_file>`_ format.
Parameters are grouped into sections. The roles of each section and
their associated parameters are described below. Note that both ``;``
and ``#`` may be used as comment characters.  Additionally, all
parameter values support environment variable expansion.

Backends
--------

These sections detail how the solver will be configured for a range of
different hardware platforms. If a hardware specific backend section is
omitted, then PyFR will fall back to built-in default settings.

.. toctree::
   :maxdepth: 3

   backends/backend.rst
   backends/backend-cuda.rst
   backends/backend-hip.rst
   backends/backend-metal.rst
   backends/backend-opencl.rst
   backends/backend-openmp.rst

Systems
-------

These sections setup and control the physical system being solved, as
well as charateristics of the spatial and temporal schemes to be used.

.. toctree::
   :maxdepth: 3

   systems/constants.rst
   systems/solver.rst
   systems/solver-time-integrator.rst
   systems/solver-dual-time-integrator-multip.rst
   systems/solver-entropy-filter.rst
   systems/solver-artificial-viscosity.rst
   systems/solver-interfaces.rst
   systems/soln-filter.rst

Boundary and Initial Conditions
-------------------------------

These sections allow users to set the boundary and initial
conditions of calculations.

.. toctree::
   :maxdepth: 3

   boundary-initial-conditions/soln-bcs.rst
   boundary-initial-conditions/soln-ics.rst

Nodal Point Sets
----------------

Solution point sets must be specified for each element type that is used
and flux point sets must be specified for each interface type that is
used. If anti-aliasing is enabled then quadrature point sets for each
element and interface type that is used must also be specified. For
example, a 3D mesh comprised only of prisms requires a solution point
set for prism elements and flux point sets for quadrilateral and
triangular interfaces.

.. toctree::
   :maxdepth: 3

   nodal-point-sets/solver-interfaces-line.rst
   nodal-point-sets/solver-interfaces-quad.rst
   nodal-point-sets/solver-interfaces-tri.rst
   nodal-point-sets/solver-elements-quad.rst
   nodal-point-sets/solver-elements-tri.rst
   nodal-point-sets/solver-elements-hex.rst
   nodal-point-sets/solver-elements-tet.rst
   nodal-point-sets/solver-elements-pri.rst
   nodal-point-sets/solver-elements-pyr.rst

Plugins
-------

Plugins allow for powerful additional functionality to be swapped in and
out. There are two classes of plugin available; solution plugins which
are prefixed by ``soln-`` and solver plugins which are prefixed by
``solver-``. It is possible to create multiple instances of the same
solution plugin by appending a suffix, for example::

    [soln-plugin-writer]
    ...

    [soln-plugin-writer-2]
    ...

    [soln-plugin-writer-three]
    ...

Certain plugins also expose functionality via a CLI, which can
be invoked independently of a PyFR run.

Solution Plugins
^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 3

   plugins/soln-plugin-ascent.rst
   plugins/soln-plugin-dtstats.rst
   plugins/soln-plugin-fluid-force.rst
   plugins/soln-plugin-fwh.rst
   plugins/soln-plugin-integrate.rst
   plugins/soln-plugin-nancheck.rst
   plugins/soln-plugin-pseudostats.rst
   plugins/soln-plugin-residual.rst
   plugins/soln-plugin-sampler.rst
   plugins/soln-plugin-tavg.rst
   plugins/soln-plugin-writer.rst

Solver Plugins
^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 3

   plugins/solver-plugin-source.rst
   plugins/solver-plugin-turbulence.rst

Regions
^^^^^^^

Certain plugins are capable of performing operations on a subset of the
elements inside the domain.  One means of constructing these element
subsets is through parameterised regions.  Note that an element is
considered part of a region if *any* of its shape points are found to be
contained within the region.  A consequence of this is that a region may
not strictly enclose a shape; this can be resolved through the
`region-expand` directive. Supported regions:

Rectangular cuboid ``box(x0, x1)``
  A rectangular cuboid defined by two diametrically opposed vertices.
  Valid in both 2D and 3D.

Conical frustum ``conical_frustum(x0, x1, r0, r1)``
  A conical frustum whose end caps are at *x0* and *x1* with radii
  *r0* and *r1*, respectively. Only valid in 3D.

Cone ``cone(x0, x1, r)``
  A cone of radius *r* whose centre-line is defined by *x0* and *x1*.
  Equivalent to ``conical_frustum(x0, x1, r, 0)``. Only valid in 3D.

Cylinder ``cylinder(x0, x1, r)``
  A circular cylinder of radius *r* whose centre-line is defined by
  *x0* and *x1*. Equivalent to ``conical_frustum(x0, x1, r, r)``.
  Only valid in 3D.

Cartesian ellipsoid ``ellipsoid(x0, a, b, c)``
  An ellipsoid centred at *x0* with Cartesian coordinate axes whose
  extents in the *x*, *y*, and *z* directions are given by *a*, *b*,
  and *c*, respectively. Only valid in 3D.

Sphere ``sphere(x0, r)``
  A sphere centred at *x0* with a radius of *r*. Equivalent to
  ``ellipsoid(x0, r, r, r)``. Only valid in 3D.

STL ``stl('name')``
  An STL region.  Note that the region *name* must have been already
  added to the mesh file with ``pyfr region add``.  Only valid in 3D.
  Additionally, region itself *must* be closed.

All region also support rotation.  In 2D this is accomplished by passing
a trailing `rot=angle` argument where `angle` is a rotation angle in
degrees; for example ``box((-5, 2), (2, 0), rot=30)``.  In 3D the syntax
is `rot=(phi, theta, psi)` and corresponds to a sequence of Euler angles
in the so-called *ZYX convention*.  Region expressions can also be added
and subtracted together  arbitrarily.  For example ``box((-10, -10,
-10), (10, 10, 10)) - sphere((0, 0, 0), 3)`` will result in a
cube-shaped region with a sphere cut out of the middle.

Additional Information
----------------------

The :ref:`INI<configuration-file>` file format is very versatile. A
feature that can be useful in defining initial conditions is the
substitution feature and this is demonstrated in the
:ref:`soln-plugin-integrate` plugin example.

To prevent situations where you have solutions files for unknown
configurations, the contents of the ``.ini`` file are added as an
attribute to ``.pyfrs`` files. These files use the HDF5 format and can
be straightforwardly probed with tools such as h5dump.

In several places within the ``.ini`` file expressions may be used. As
well as the constant ``pi``, expressions containing the following
functions are supported:

#. ``+, -, *, /`` --- basic arithmetic

#. ``sin, cos, tan`` --- basic trigonometric functions (radians)

#. ``asin, acos, atan, atan2`` --- inverse trigonometric functions

#. ``exp, log`` --- exponential and the natural logarithm

#. ``tanh`` --- hyperbolic tangent

#. ``pow`` --- power, note ``**`` is not supported

#. ``sqrt`` --- square root

#. ``abs`` --- absolute value

#. ``min, max`` --- two variable minimum and maximum functions,
   arguments can be arrays
