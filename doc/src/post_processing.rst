.. _ParaViewPar: https://docs.paraview.org/en/latest/ReferenceManual/parallelDataVisualization.html

***************
Post Processing
***************

The following sections contain best practices for post processing a
simulation using ParaView.

High-Order Export
-----------------

By default PyFR will output a high-order VTK file.  However, to take
full advantage of this functionality it is necessary to adjust the
*Nonlinear Subdivision Level* property in ParaView.  This property is
configured on a per-filter basis and defaults to one.  This results in
each high-order element being internally subdivided once by ParaView
prior to further processing.  While this is often a reasonable first
approximation it is typically inadequate for moderate-order simulations.
It may therefore be necessary to move the slider to two or three levels
of subdivision.  Note, however, that ParaView subdivides *recursively*
and hence a quadrilateral subject to two levels of subdivision is broken
down into 16 elements.  This is in contrast to PyFR where, when
employing subdivision, ``--eopt=divisor:2`` results in each
quadrilateral being divided into four elements.

Clean to Grid
-------------

By default, volume, boundary, and spanwise VTK exports are written on a
continuous grid: duplicate vertices within each piece are removed and
point data at shared vertices is averaged.  For ``.pvtu`` outputs, each
piece remains self-contained, but partition-boundary duplicates carry
the same averaged values on each side.  Thus, in most cases no *Clean to
Grid* filter is needed after export.

To emit the discontinuous FR representation instead, pass
``--discontinuous`` to ``pyfr export volume``, ``pyfr export boundary``,
or ``pyfr export spanwise``.  When visualising such discontinuous
outputs in ParaView, run *Clean to Grid* and set the *Point Data
Weighting Strategy* to *Average by Number*.

Tessellate
----------

When working with high-order outputs an alternative to the  *Clean to
Grid* filter is the *Tessellate* filter.  This takes a high-order grid
with potentially discontinuous field values and *adaptively* subdivides
it into a larger number of linear elements.  Here, there are two primary
parameters: *Chord Error* which controls the amount of permitted
geometrical error and the unnamed *Field Error* list-box which controls
the amount of permitted error in the field values themselves.  Although
not obvious from the user interface, it is possible to provide a
separate error tolerance for each field.  Secondary parameters include
the *Maximum Number of Subdivisions* and *Merge Points* with the latter
subsuming the functionality of *Clean to Grid*.  Note that this filter
can be **extremely computationally expensive**, especially when tight
error tolerances are chosen.

Avoiding Seams
--------------

When working with mixed element meshes or with .pvtu files obtained by
running pyfr export in parallel, it is possible for seams to appear in
some ParaView filters.  These can be avoided by adding the *Ghost Cells*
plugin after loading the dataset (or after *Tessellate* when using that
filter).

Parallel Processing
-------------------

When post-processing large data sets it is important to run ParaView in
parallel.  This can be done either interactively using *pvserver* or in
a batch capacity using *pvbatch*.  Full details can be found in the
`ParaView documentation <ParaViewPar_>`_.  Upon opening a file the
recommended filter stack is as follows:

#. *Redistribute Dataset* which will distribute the cells such that each
   ParaView rank has a roughly equivalent amount of data.
#. *Tessellate* if adaptive subdivision of high-order data is required.
#. *Ghost Cells* to avoid seams in filters that use neighbouring cells.

When working with extremely large datasets in parallel it is recommended
to use the .pvtu file format.  This has the advantage of being already
partitioned such that there is no need to run the *Redistribute Dataset*
filter.  Here, the number parts in the .pvtu file is equal to the number
of ranks *pyfr export* is run with.  For optimal performance, the number
of parts should be an integer multiple of the number of ranks *pvserver*
or *pvbatch* will be run with.   Furthermore, it is important that each
of these ranks have a similar number of elements.  This is not usually
an issue except when *pyfr export* is using a weighted partitioning or
when the .pyfrs files are subset.  In the former case the solution is to
add a uniform partitioning to the mesh and employ this for the export.
In the latter case the only robust solution is to use the *Redistribute
Dataset* to rebalance the file.

Spanwise Average Export
-----------------------

``pyfr export spanwise`` reduces a 3D solution to a 2D unstructured VTU
by averaging along one Cartesian axis.  The user specifies a **pair of
surfaces** that bound the averaging direction: either a periodic pair
or two named boundaries.

.. code-block:: shell

    pyfr export spanwise mesh.pyfrm tavg.pyfrs out.vtu --eopt periodic:0
    pyfr export spanwise mesh.pyfrm tavg.pyfrs out.vtu --eopt boundary:bcwalllower,bcwallupper

The averaging axis is derived automatically from the pair geometry:
for periodic pairs from the translation vector *T*; for boundary
pairs from the centroid displacement between the two surfaces.

For structured ``hex``/``pri`` meshes the exporter uses exact
quadrature along element columns; for unstructured meshes it falls
back to trapezium-rule sampling at ``nstations`` positions.  The
path is selected automatically.

Options (passed via ``--eopt key:value``):

- ``periodic:NAME`` -- use periodic pair ``periodic/NAME``.  The
  translation vector must be axis-aligned.  Mutually exclusive
  with ``boundary:``.
- ``boundary:LO,HI`` -- use two named boundary groups as the pair,
  e.g. ``boundary:bcwalllower,bcwallupper``.  The two surfaces
  must be separated along a single Cartesian axis.  Mutually
  exclusive with ``periodic:``.
- ``nstations:N`` -- number of sampling stations (sampled path
  only).  Defaults to ``4*(p+1)`` where ``p`` is the solver order.

The output VTU is 2D (the averaging-axis component is set to zero).
All input scalar fields are preserved as VTK PointData arrays.

Boundary and STL Export
-----------------------

When working with surface data from a boundary export, it is
recommended to start with the following filter pipeline:

#. *Extract Surface* which will convert the internal representation of
   the surface from an unstructured grid to polygonal data.
#. *Generate Surface Normals* which will yield a much smoother surface.
   The normal data is also needed for computing quantities such as the
   skin-friction coefficient :math:`C_f`.

To reduce the computational cost associated with exporting boundary data
it is recommended to use *regions*.  For example, if our ultimate goal
is to analyse data on a boundary called *wall* then we can configure our
solution writer as:

.. code-block:: ini

    [soln-plugin-writer]
    ...
    write-gradients = true
    region = wall

which will output the solution and gradient data for the elements on our
boundary.  Then, we can pass this subset solution file to ``pyfr export
boundary``.  Alternatively, if our goal is to export an STL region
called *rgn* then we can configure our solution writer as:

.. code-block:: ini

    [soln-plugin-writer]
    ...
    write-gradients = true
    region = stl('rgn')
    region-expand = 1
    region-type = surface

which will write out approximately three layers of elements around the
surface defined by our STL region.  The need for multiple layers is due
to the coarse-grained nature of the region code and ensures that the
subsequent export step will have enough elements to work with.
