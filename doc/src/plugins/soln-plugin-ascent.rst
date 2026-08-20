.. _soln-plugin-ascent:

[soln-plugin-ascent]
^^^^^^^^^^^^^^^^^^^^

Uses `Alpine Ascent <https://github.com/Alpine-DAV/ascent>`_ to plot
on-the-fly.  The following parameters can then be set:

#. ``nsteps`` --- produce the plots every ``nsteps`` time steps:

    *int*

#. ``division`` --- the level of linear subdivison to use

    *int*

#. ``region`` --- volume region to be written, specified as either
   the entire domain using ``*``, a combination of the geometric
   shapes specified in :ref:`user_guide:regions`, or a sub-region of
   elements that have faces on a specific domain boundary via the
   name of the domain boundary.  Only applies to the volume source:

    ``*`` | ``shape(args, ...)`` | *string*

#. ``region-expand`` --- how many layers to grow the volume region by:

    *int*

#. ``clean`` --- if true, average coincident subdivided vertices on
   shared element faces, edges, and corners (and across MPI partition
   boundaries).  Applies to both volume and surface sources.
   Eliminates the faint cell-interface facets that can be visible at
   high subdivision levels, at the cost of one extra collective at
   startup and a per-field averaging step at every render.  Default
   is true, matching ``pyfr export``; set to false to recover the
   per-element discontinuous behaviour:

    *bool*

#. ``surface-{name}`` --- attach a named surface source to the scene,
   sampling fields on the specified boundary.  The value is the
   boundary name (with optional ``bc/`` prefix); a brace enumeration
   such as ``bc/{le, te}`` fuses several boundaries into one source.
   When at least one
   ``surface-{name}`` is set, the volume source is omitted unless
   ``volume = true`` is also given.  Multiple surface sources can be
   defined and used in the same scene:

    *string*

#. ``volume`` --- if true, include the volume source alongside any
   ``surface-{name}`` sources.  Default is false when surfaces are
   defined and true otherwise:

    *bool*

#. ``basedir`` --- directory in which to write rendered images.
   Resolved relative to the current working directory.  Default is
   ``.``:

    *string*

#. ``viskores-backend`` --- Viskores (formerly VTK-m) execution backend
   used by Ascent.  Common values are ``serial``, ``openmp``, ``cuda``
   and ``kokkos``; availability depends on how Ascent was built.
   Default is ``serial``:

    *string*

The plugin's data model is a graph: fields flow into pipelines and
plots, plots are grouped into scenes, and scenes are drawn by one or
more renders.  Each concept is defined below.

Field
    A scalar or vector quantity defined by a mathematical expression
    over the primitive variables.  Each field is published per source
    under the namespaced name ``{source}_{field}``.

Pipeline
    A chain of Ascent filters (``vector_magnitude``, ``contour``,
    ``slice``, ``qcriterion``, ...) that transforms fields into new
    fields or geometry.  Filters reference fields by their fully
    namespaced name.

Plot
    A single visual element inside a scene.  Specifies what to draw
    (``type``), which ``field`` to colour by, and optionally which
    ``pipeline`` to consume.

Scene
    A collection of plots that share a coordinate system and end up
    composited into the same image.

Render
    A camera and image configuration attached to a scene.  Each render
    produces one PNG per invocation; multiple renders give multiple
    views of the same plots.

#. ``field-{name}`` --- this is an extension to the Ascent library where
   users define expressions for the fields used. This can either be a
   scalar or a vector, where the latter is defined by a comma separted
   list of expressions.

    *string* | *string*, *string* (, *string*)

#. ``scene-{name}`` --- a scene to plot with Ascent options passed in
   a dictionary.  The scene must define ``plots`` as a list of plot
   dictionaries.  Each plot needs a ``field`` (or a ``pipeline`` whose
   output is selected via ``field``) defined via ``field-{name}`` or a
   pipeline.  When more than one source is configured, each plot must
   set ``source`` to the desired source name (``volume`` or the name
   from a ``surface-{name}`` entry).  One or multiple
   ``render-{name}`` dictionaries configure the rendering; a single
   default render is created if a scene-level ``image-name`` or
   ``image-prefix`` is supplied with no explicit ``render-{name}``:

    *dict*

#. ``pipeline-{name}`` --- a pipeline of data manipulations that can be
   used within a scene. The value is a dictionary containing the valid
   configuration options. Pipeline objects can be stacked together to
   form a pipeline of filters by making a list of dictionaries.
   Pipeline filters reference fields by their fully namespaced name
   (e.g. ``field: volume_velocity``); the q-criterion and vorticity
   filters both require an explicit ``field`` parameter.

   *dict* | [*dict*]

#. ``postproc-{name}`` --- run a PyFR post-processing plugin in-situ
   on one or more sources, publishing its output as a derived field.
   Value is a comma-separated list of source names (``volume`` and/or
   any ``surface-{name}``).  See
   :ref:`developer_guide:Post-processing Plugins` for the available
   plugins and their configuration.  Fields are namespaced per source
   like any other field (``volume_mach``, ``airfoil_yplus``).
   Postprocs that need gradients (``vorticity``, ``yplus``, ``cf``)
   work directly in-situ; via ``pyfr ascent render`` they need a
   snapshot carrying gradients (``write-gradients = true`` for
   ``soln`` snapshots, matching ``avg-grad_{var}_{dim}`` entries for
   ``tavg``):

   *string* (, *string*)

When rendering a ``tavg`` snapshot, only ``cp`` and ``vorticity``
give the true time average; the others give the postproc of the
averaged state.

Example:

.. code-block:: ini

    [soln-plugin-ascent]
    nsteps = 200
    division = 5

    field-kenergy = 0.5*rho*(u*u + v*v)
    scene-ke = {'plots': [{'type': 'pseudocolor', 'field': 'kenergy'}], 'render-1': {'image-name': 'ke-{t:.1f}'}}

    field-mom = rho*u, rho*v
    pipeline-amom = {'type': 'vector_magnitude', 'field': 'volume_mom', 'output-name': 'mag'}
    scene-va = {'plots': [{'type': 'pseudocolor', 'pipeline': 'amom', 'field': 'mag'}], 'render-1': {'image-width': 128, 'image-name': 'm1-{t:4.2f}'}, 'render-2': {'image-width': 256, 'image-name': 'm2-{t:4.2f}'}}

Hybrid example combining a surface plot of pressure on the airfoil
boundary with volume Q-criterion isosurfaces coloured by velocity
magnitude:

.. code-block:: ini

    [soln-plugin-ascent]
    nsteps = 200
    division = 3
    volume = true
    surface-airfoil = bc/airfoil

    field-velocity = u, v, w
    field-pressure = p
    field-vmag = sqrt(u*u + v*v + w*w)

    pipeline-qiso = [{'type': 'qcriterion', 'field': 'volume_velocity', 'output-name': 'qc'}, {'type': 'contour', 'field': 'qc', 'iso-values': [1.0, 5.0]}]

    scene-hybrid = {'plots': [{'type': 'pseudocolor', 'field': 'pressure', 'source': 'airfoil'}, {'type': 'pseudocolor', 'pipeline': 'qiso', 'field': 'vmag', 'source': 'volume'}], 'render-1': {'image-name': 'hybrid-{t:.2f}'}}

Postproc example - rendering surface y+ and Cp on a wing alongside the
volume Mach number, without writing the post-processing expressions by
hand:

.. code-block:: ini

    [constants]
    rho-inf = 1.0
    u-inf = 1.0
    p-inf = 31.74603174603175

    [soln-plugin-ascent]
    nsteps = 100
    division = 3
    volume = true
    surface-wing = bc/wing

    postproc-mach = volume
    postproc-cp = wing
    postproc-yplus = wing

    scene-mach = {'plots': [{'type': 'pseudocolor', 'field': 'mach', 'source': 'volume'}], 'render-1': {'image-name': 'mach-{t:.2f}'}}
    scene-cp = {'plots': [{'type': 'pseudocolor', 'field': 'cp', 'source': 'wing'}], 'render-1': {'image-name': 'cp-{t:.2f}'}}
    scene-yplus = {'plots': [{'type': 'pseudocolor', 'field': 'yplus', 'source': 'wing'}], 'render-1': {'image-name': 'yplus-{t:.2f}'}}

User-defined field expressions are always namespaced per source as
``{source}_{field}`` (e.g. ``volume_velocity``, ``airfoil_pressure``).
Plots auto-resolve via the plot's ``source`` key, so users typically
write the bare field name in a plot.  Pipeline filters have no source
context and must reference fields by their fully namespaced name (e.g.
``field: volume_velocity``); this includes ``qcriterion`` and
``vorticity``, which both require an explicit ``field`` parameter.
Pipeline outputs (``output-name``) are left un-namespaced and can be
referenced verbatim downstream.

Note that setting ``nsteps`` to be too small can have a significant
impact on performance as generating each image has overhead and may
require some MPI communication to occur.

This plugin also exposes functionality via a CLI. The following functions
are available

- ``pyfr ascent render`` --- render an image from a pre-existing mesh
  and one or more solution files. It must be run with the same number
  of ranks as partitions in the mesh. By default it will use settings
  from the first section of the settings file which defines a scene;
  as such a simulation configuration file with a
  ``[soln-plugin-ascent]`` section can be passed directly.
  Alternatively, a specific section name can be provided. In both
  cases all other sections are ignored.  Multiple solution files can
  be passed; the renderer is rebuilt when the embedded solver
  configuration changes between files, so mixed solver orders and a
  mix of ``soln``/``tavg`` snapshots are supported in one invocation.

  Time-averaged ``tavg`` snapshots are detected automatically and the
  mapping from tavg field names to canonical primitive variables
  (``rho``, ``u``, ``v``, ``w``, ``p``, and the corresponding
  ``grad_{var}_{x,y,z}`` entries when present) is inferred from the
  tavg configuration embedded in the snapshot.

  Example:

  .. code-block:: ini

      pyfr ascent render mesh.pyfrm solution.pyfrs settings.ini
