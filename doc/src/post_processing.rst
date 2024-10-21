.. highlight:: none

***************
Post Processing
***************

The following sections contain best pratices for post processing a
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
It may therefore be necessary to mode the slider to two or three levels
of subdivision.  Note, however, that ParaView subdivides *recursively*
and hence a quadriteral subject to two levels of subdivision is broken
down into 16 elements.  This is in contrast to PyFR where, when
employing subdivision, ``--eopts=divisor:2`` results in each
quadrilateral being divided into four elements.

Clean to Grid
-------------

Upon opening an exported file in ParaView one should *always* run the
*Clean to Grid* filter.  This will eliminate duplicate vertices along
the faces and edges between elements that arise as a consequence of the
discontinous nature of the FR approach.  Running this filter will not
only result in cleaner visuals but will also improve the performance of
ParaView.

Boundary and STL Export
-----------------------

When working with surface data, be it from a boundary or an STL file, it
is recommended to start with the following filter pipeline:

#. *Clean to Grid* for the reasons outlined above.
#. *Extract Surface* which will convert the internal representation of
   the surface from an unstructured grid to polygonal data.
#. *Generate Surface Normals* which will yield a much smoother surface.
   The normal data is also needed for computing quantities such as the
   skin-friction coefficient :math:`C_f`.

To reduce the computational cost associated with exporting boundary data
it is recommended to use *regions*.  For example, if our ultimate goal
is to analyse data on a boundary called *wall* then we can configure our
solution writer as::

    [soln-plugin-writer]
    ...
    write-gradients = true
    region = wall

which will output the solution and gradient data for the elements on our
boundary.  Then, we can pass this subset solution file to ``pyfr export
boundary``.