.. _soln-plugin-ascent:

[soln-plugin-ascent]
^^^^^^^^^^^^^^^^^^^^

Uses `Alpine Ascent <https://github.com/Alpine-DAV/ascent>`_ to plot
on-the-fly.  The following parameters can then be set:

#. ``nsteps`` --- produce the plots every ``nsteps`` time steps:

    *int*

#. ``division`` --- the level of linear subdivison to use

    *int*

#. ``region`` --- region to be written, specified as either the
   entire domain using ``*``, a combination of the geometric shapes
   specified in :ref:`regions`, or a sub-region of elements that have
   faces on a specific domain boundary via the name of the domain
   boundary:

    ``*`` | ``shape(args, ...)`` | *string*

#. ``region-expand`` --- how many layers to grow the region by:

    *int*

There are then three components that can be used to build plots. Scenes
which define the render, Pipelines that can be used to apply filers and
build sequenceces of data manipulations, and Fields which are used to
define field expresions.

#. ``field-{name}`` --- this is an extension to the Ascent library where
   users define expressions for the fields used. This can either be a
   scalar or a vector, where the latter is defined by a comma separted
   list of expressions.

    *string* | *string*, *string* (, *string*)

#. ``scene-{name}`` --- a scene to plot with Ascent options passed in a
   dictionary. Each scene needs a field, and the expression for that
   field must have been set either via a field command or a pipeline.
   Additionally, one or multiple ``render-{name}`` dictionaries must be
   defined to configure the rendering of the scene. Multiple render
   dictionaries give multiple views of the same scene.

    *dict*

#. ``pipeline-{name}`` --- a pipeline of data manipulations that can be
   used within a scene. The value is a dictionary containing the valid
   configuration options. Pipeline objects can be stacked together to
   form a pipeline of filters by making a list of dictionaries.
   Finally, the q-criterion and vorticity filters require that a field
   called velocity is defined.

   *dict* | [*dict*]

Example::

    [soln-plugin-ascent]
    nsteps = 200
    division = 5

    field-kenergy = 0.5*rho*(u*u + v*v)
    scene-ke = {'render-1': {'image-name': 'ke-{t:.1f}'}, 'field': 'kenergy', 'type': 'pseudocolor'}

    field-mom = rho*u, rho*v
    pipeline-amom = {'type': 'vector_magnitude', 'field': 'mom', 'output-name': 'mag'}
    scene-va = {'type': 'pseudocolor', 'pipeline': 'amom', 'field': 'mag', 'render-1': {'image-width': 128, 'image-name': 'm1-{t:4.2f}'}, 'render-2': {'image-width': 256, 'image-name': 'm2-{t:4.2f}'}}

Note that setting ``nsteps`` to be too small can have a significant
impact on performance as generating each image has overhead and may
require some MPI communication to occur.

This plugin also exposes functionality via a CLI. The following functions
are available

- ``pyfr ascent render`` --- render an image from a pre-existing mesh
  and solution file. It must be run with the same number of ranks as
  partitions in the mesh. By default it will use settings from the first
  section of the settings file that it is passed. Alternatively, a
  specific section name can be provided. In both cases all other
  sections are ignored.

  Example::

    pyfr ascent render mesh.pyfrm solution.pyfrs settings.ini
