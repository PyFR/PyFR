*******
Meshing
*******

Importing Meshes
================

PyFR is capable of importing meshes in the Gmsh .msh format;
specifically versions 2.2 and 4.1 of the ASCII format with the latter
being recommended. Although these files can include partitioning
information this is ignored by PyFR. In order to be imported by PyFR
the following guidelines must be followed:

* The mesh must contain at least one physical group at the *volume*
  dimension (the top dimension of the mesh: Physical Volume in 3D,
  Physical Surface in 2D).  The name of such a group is not
  constrained and every volume physical group becomes a *region tag*
  on the elements it contains.  Multiple volume groups may be defined
  and a single element may belong to more than one region.  Region
  tags can subsequently be used for
  :ref:`tag-weighted partitioning <perf region tags>` and referenced
  in plugin region expressions of the form ``tag/<name>``.
* Boundaries can be assigned any physical name. However, the node
  numbers of faces on the boundary *must* match perfectly with the
  faces of the corresponding volume elements. Any discrepancies here
  will result in ``KeyError`` exceptions being raised on import. Such
  errors usually arise when Gmsh is asked to make a volume mesh whose
  characteristic size near a boundary is greater than the spacing of
  points used to define the boundary itself. The result is a boundary
  mesh which is inconsistent with the surface of the volume mesh.
* Periodic boundary conditions can be defined by assigning one of the
  boundaries a physical name of "periodic-*name*-l" and the other a
  physical name of "periodic-*name*-r" where *name* is an arbitrary
  identifier, with the periodic relationship defined via a
  ``$Periodic`` section (MSH 4.1). The ``$Periodic`` section is not
  strictly necessary for translation periodicity. Note that only
  translational periodicity is currently supported.
* Curved elements are supported up to quartic order. Such elements must
  be *complete* Lagrange elements as opposed to *incomplete* serendipity
  elements.
