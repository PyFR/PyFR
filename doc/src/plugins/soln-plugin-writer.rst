********************
[soln-plugin-writer]
********************

Periodically write the solution to disk in the pyfrs format.
Parameterised with

#. ``dt-out`` --- write to disk every ``dt-out`` time units:

    *float*

#. ``basedir`` --- relative path to directory where outputs will be
   written:

    *string*

#. ``basename`` --- pattern of output names:

    *string*

#. ``write-gradients`` --- if to write out gradient data:

    *bool*

#. ``async-timeout`` --- how long asynchronous file writes are allowed
   to take before becoming blocking:

    *float*

#. ``post-action`` --- command to execute after writing the file:

    *string*

#. ``post-action-mode`` --- how the post-action command should be
   executed:

    ``blocking`` | ``non-blocking``

#. ``region`` --- region to be written, specified as either the
   entire domain using ``*``, a combination of the geometric shapes
   specified in :ref:`regions`, or a sub-region of elements that have
   faces on a specific domain boundary via the name of the domain
   boundary:

    ``*`` | ``shape(args, ...)`` | *string*

#. ``region-type`` --- if to write all of the elements contained inside
   the region or only those which are on its surface:

    ``volume`` | ``surface``

#. ``region-expand`` --- how many layers to grow the region by:

    *int*

Example::

    [soln-plugin-writer]
    dt-out = 0.01
    basedir = .
    basename = files-{t:.2f}
    async-timeout = 0
    post-action = echo "Wrote file {soln} at time {t} for mesh {mesh}."
    post-action-mode = blocking
    region = box((-5, -5, -5), (5, 5, 5))
    region-type = volume
    region-expand = 0
