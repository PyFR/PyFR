.. highlight:: none

*******
Meshing
*******

Importing Meshes
^^^^^^^^^^^^^^^^

PyFR is capable of importing meshes in the Gmsh .msh format;
specifically versions 2.2 and 4.1 of the ASCII format with the latter
being recommended. Although these files can include partitioning
information this is ignored by PyFR. In order to be imported by PyFR
the following guidelines must be followed:

* Elements must be assigned a *physical name* of "fluid".
* Boundaries can be assigned any physical name. However, the node
  numbers of faces on the boundary *must* match perfectly with the
  faces of the corresponding elements in the fluid region. Any
  discrepancies here will result in ``KeyError`` exceptions being
  raised on import. Such errors usually arise when Gmsh is asked to
  make a volume mesh whose characteristic size near a boundary is
  greater than the spacing of points used to define the boundary
  itself. The result is a boundary mesh which is inconsistent with the
  surface of the fluid mesh.
* Periodic boundary conditions can be defined by assigning one of the
  boundaries a physical name of "periodic-*name*-l" and the other a
  physical name of "periodic-*name*-r" where *name* is an arbitrary
  identifier. Note that only translational peridocity is currently
  supported.
* Curved elements are supported up to quartic order. Such elements must
  be *complete* Lagrange elements as opposed to *incomplete* serendipity
  elements.
