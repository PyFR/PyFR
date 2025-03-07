File Format
===========

The PyFR mesh and solution file formats are based around HDF5. All
arrays are stored using little-endian byte-ordering. Floating point data
can be either single or double precision. Strings are represented as
fixed-size ``H5T_STRING`` datasets with either ASCII or UTF-8 encoding.
Certain records are stored as serialised INI files.

For this tutorial we will make use of the 2D incompressible cylinder
test case which ships with PyFR. This is a mixed element case containing
quadratically curved elements and a range of boundaries. We begin by
importing the mesh and adding a partitioning::

   $ pyfr import inc-cylinder.msh inc-cylinder.pyfrm
   $ pyfr partition add inc-cylinder.pyfrm 3 -equad:2 -etri:1 -pmetis

Mesh Format
-----------

Inspecting the structure of our mesh we find::

   /                        Group
   /codec                   Dataset {12}
   /creator                 Dataset {SCALAR}
   /eles                    Group
   /eles/quad               Dataset {196}
   /eles/tri                Dataset {3231}
   /mesh-uuid               Dataset {SCALAR}
   /nodes                   Dataset {7345}
   /partitionings           Group
   /partitionings/1         Group
   /partitionings/1/eles    Dataset {3427}
   /partitionings/3         Group
   /partitionings/3/eles    Dataset {3427}
   /partitionings/3/neighbours Dataset {4}
   /version                 Dataset {SCALAR}

The ``/creator`` dataset is a string corresponding to the program which
created the file. This is usually ``pyfr vX.Y.Z`` where X, Y, and Z,
correspond to the major, minor, and patch versions. The ``/version`` is
an integer which contains the specific revision of the format; currently
it is required to be 1. The ``/mesh-uuid`` is a hex-encoded unique
identifier which is derived from hashing the nodes and elements of a
mesh. Whenever a solution is written the ``/mesh-uuid`` from the current
mesh is copied over. As such it can be used to check that a solution is
indeed associated with a particular mesh.

The elements of the mesh are defined in the ``eles`` group. Each
distinct element type present in the mesh is given its own dataset.
Hence, in the above example we see that the mesh in question has 196
``quad`` elements and 3231 ``tri`` elements.

Inspecting the structure of the ``eles/quad`` dataset we find::

   DATASET "/eles/quad" {
      DATATYPE  H5T_COMPOUND {
         H5T_ARRAY { (9) H5T_STD_I64LE } "nodes";
         H5T_ENUM {
            H5T_STD_I8LE;
            "FALSE"            0;
            "TRUE"             1;
         } "curved";
         H5T_ARRAY { (4) H5T_COMPOUND {
            H5T_STD_I16LE "cidx";
            H5T_STD_I64LE "off";
         } } "faces";
      }
      DATASPACE  SIMPLE { ( 196 ) / ( 196 ) }
   }

The ``nodes`` field defines the *node numbers* which specify the shape
points of each quad. Each node number is an index into the ``/nodes``
dataset. The degree of curvature for each element type is inferred from
the length of this array. In the above we conclude that our elements are
quadratically curved on account of the array having a length of 9. The
``curved`` field is a convenience member which can be used to quickly
determine if a particular element is actually curved or not. Finally,
the ``faces`` field contains information about the connectivity of each
face. The length of this array is given by the number of faces on the
element; in the above case it is four. Each element of this array is
made up of two pieces of information: a ``cidx`` number and an ``off``.
The ``cidx`` field is an entry into the ``/codec`` array and is used to
determine *what* the face is connected to. Inspecting the ``/codec``
array for our mesh we find::

   DATASET "/codec" {
      DATATYPE  H5T_STRING {
         STRSIZE 11;
         STRPAD H5T_STR_NULLPAD;
         CSET H5T_CSET_ASCII;
         CTYPE H5T_C_S1;
      }
      DATASPACE  SIMPLE { ( 12 ) / ( 12 ) }
      DATA {
      (0): "eles/tri\000\000\000", "eles/tri/0\000", "eles/tri/1\000",
      (3): "eles/tri/2\000", "eles/quad\000\000", "eles/quad/0", "eles/quad/1",
      (7): "eles/quad/2", "eles/quad/3", "bc/wall\000\000\000\000",
      (10): "bc/inlet\000\000\000", "bc/outlet\000\000"
      }
   }

This is always an array of null-padded strings. From this we see that
``/codec[1] = "eles/tri/0"`` and thus a ``cidx`` value of 1 corresponds
to face 0 of a triangle The element number is given by the ``off``
member. For example, here ``/eles/quad[98].faces[0] = (3, 334)`` which
means that face 0 of quad 98 is connected to face 2 of triangle 334.
Correspondingly, ``/eles/tri[334].faces[2] = (5, 98)`` with
``/codec[5] = "eles/quad/0"``. When the ``cidx`` is that of a boundary
the ``off`` field is unnecessary and is guaranteed to be -1. Boundary
names are, in general, arbitrary.

Inspecting the ``/nodes`` array we have::

   DATASET "/nodes" {
      DATATYPE  H5T_COMPOUND {
         H5T_ARRAY { (2) H5T_IEEE_F64LE } "location";
         H5T_STD_U16LE "valency";
      }
      DATASPACE  SIMPLE { ( 7345 ) / ( 7345 ) }
   }

Here, each record consists of two fields: a ``location`` array which
gives the position of the node in space, and a ``valency`` number which
notes how many elements share this node. The dimension of the location
array is equal to the number of spatial dimensions in the mesh; in our
case this is two.

Node ordering
~~~~~~~~~~~~~

All nodes are specified with regards to a standard element. These
elements are:

+-------------------------+--------------------------------------------+
| Element                 | Shape points                               |
+=========================+============================================+
| Tri                     | (-1, -1), (1, -1), (-1, 1)                 |
+-------------------------+--------------------------------------------+
| Quad                    | (-1, -1), (1, -1), (-1, 1), (-1, -1)       |
+-------------------------+--------------------------------------------+
| Hex                     | (-1, -1, -1), ( 1, -1,-1), (-1, 1, -1), (  |
|                         | 1, 1, -1), (-1, -1, 1), ( 1, -1, 1), (-1,  |
|                         | 1, 1), ( 1, 1, 1)                          |
+-------------------------+--------------------------------------------+
| Pri                     | (-1, -1, -1), (1, -1, -1), (-1, 1, -1),    |
|                         | (-1, -1, 1), ( 1, -1, 1), (-1, 1, 1)       |
+-------------------------+--------------------------------------------+
| Pyr                     | (-1, -1, -1), (1, -1, -1), (-1, 1, -1),    |
|                         | (1, 1, -1), (0, 0, 1)                      |
+-------------------------+--------------------------------------------+
| Tet                     | (-1, -1, -1), (1, -1, -1), (-1, 1, -1),    |
|                         | (-1, -1, 1)                                |
+-------------------------+--------------------------------------------+

The ordering of the shape points is such that the x-axis counts
quickest, followed by the y-axis, and then finally the z-axis.
Higher-order elements are always of the Lagrange type and correspond to
equi-spaced subdivisions of the first-order standard elements.

Face numbering
~~~~~~~~~~~~~~

Face numbering is established through consideration of the
outward-facing normal vector for each face of a standard element in a
right-hand coordinate system.

+-----------------------------------+-----------------------------------+
| Element type                      | Face normals                      |
+===================================+===================================+
| Tri                               | (0, -1), (1, 1), (-1, 0)          |
+-----------------------------------+-----------------------------------+
| Quad                              | (0,-1), (1, 0), (0, 1), (-1, 1)   |
+-----------------------------------+-----------------------------------+
| Hex                               | (0, 0, -1), (0, -1, 0), (1, 0,    |
|                                   | 0), (0, 1, 0), (-1, 0, 0), (0, 0, |
|                                   | 1)                                |
+-----------------------------------+-----------------------------------+
| Pri                               | (0, 0, -1), (0, 0, 1), (0, -1,    |
|                                   | 0), (1, 1, 0), (-1, 0, 0)         |
+-----------------------------------+-----------------------------------+
| Pyr                               | (0, 0, -1), (0, -1, 0.5), (1, 0,  |
|                                   | 0.5), (0, 1, 0.5), (-1, 0, 0.5)   |
+-----------------------------------+-----------------------------------+
| Tet                               | (0, 0, -1), (0, -1, 0), (-1, 0,   |
|                                   | 0), (1, 1, 1)                     |
+-----------------------------------+-----------------------------------+

Partitioning
~~~~~~~~~~~~

Every mesh contains one of more named *partitionings*. These are used to
specify how elements of a mesh should be distributed between MPI ranks.
Each partitioning is a sub-group of the ``/partitionings`` group. For
non-trivial partitionings this group will contain two integer-array
datasets: ``eles`` and ``neighbours``.

The length of the ``eles`` dataset is *always* equal to the total number
of elements in the mesh. To interpret these element numbers it is
necessary to consult the ``regions`` attribute. This is a two
dimensional dataset where the number of rows is equal to the number of
partitions and the number of columns is equal to the number of distinct
element types plus one. For ``/partitionings/3/eles`` we have::

   ATTRIBUTE "regions" {
      DATATYPE  H5T_STD_I64LE
      DATASPACE  SIMPLE { ( 3, 3 ) / ( 3, 3 ) }
      DATA {
      (0,0): 0, 0, 1207,
      (1,0): 1207, 1207, 2415,
      (2,0): 2415, 2611, 3427
      }
   }

The numbers correspond to offsets in the ``eles`` array. To use this
array we begin by alphabetically sorting the element types in our mesh.
The starting offset for the elements of sorted index *i* in partition
*p* is ``eles.regions[p, i]`` and the ending offset is
``eles.regions[p, i+1]``.

In our example there are two element types in the mesh: *quad* and
*tri*. Hence, the ``quad`` element numbers for partition 2 are
``eles[2415:2611]`` while the ``tri`` element numbers are between
``eles[2611:3427]``. It is also immediately clear from this that neither
partition 0 or 1 contain any quad elements since their starting and
ending offsets are the same.

The ``neighbours`` dataset is a representation of the connectivity
*between* partitions. As with ``eles`` to interpret this dataset it is
necessary to consult the ``regions`` attribute. This is a one
dimensional array of offsets whose length is equal to the number of
partitions plus one. The connectivity information for partition ``p`` is
between ``neighbours.regions[p]`` and ``neighbours.regions[p+1]``. For
``/partitionings/3/neighbours`` we have::

   ATTRIBUTE "regions" {
      DATATYPE  H5T_STD_I64LE
      DATASPACE  SIMPLE { ( 4 ) / ( 4 ) }
      DATA {
      (0): 0, 1, 3, 4
      }
   }

The connectivity for partition 0 is hence given by ``neighbours[0:1]``
while the connectivity for partition 1 is given by ``neighbours[1:3]``.
Just by looking at this array we conclude that partitions 0 and 2 only
have a single neighbouring partition, whilst partition 1 has two
neighbours.

Solution Format
---------------

The general structure of a solution file is::

   /                        Group
   /config                  Dataset {SCALAR}
   /config-0                Dataset {SCALAR}
   /creator                 Dataset {SCALAR}
   /mesh-uuid               Dataset {SCALAR}
   /soln                    Group
   /soln/p3-quad            Dataset {196, 3, 16}
   /soln/p3-quad-parts      Dataset {196}
   /soln/p3-tri             Dataset {3231, 3, 10}
   /soln/p3-tri-parts       Dataset {3231}
   /stats                   Dataset {SCALAR}
   /version                 Dataset {SCALAR}

The ``/creator``, ``/mesh-uuid``, and ``/version`` datasets have
identical meanings to those in the mesh file format. When opening a
solution it is important to check that the UUID matches that of the
associated mesh.

To obtain the path to the solution data it is necessary to consult the
``/stats`` dataset. This is a serialised INI file which contains
information about the solution. Of interest to us is the ``[data]``
section::

   [data]
   fields = p,u,v
   prefix = soln

Here, the ``fields`` key contains the names of the field variables in
the solution. As this particular case corresponds to incompressible
Navier–Stokes equations there are 3 field variables: pressure and two
velocities denoted by *p*, *u*, and *v*, respectively. Note that the
solution may contain more fields than expected. This can happen if the
user has requested that gradient data also be output. Applications
should *not* depend on the ordering of field variables. The ``prefix``
key tells us which group contains the solution data itself. Usually, the
prefix is either *soln* for solutions or *tavg* for time-average data.

All solution data arrays have three dimensions: the first corresponding
to the number of elements in the array, the second to the number of
field variables, and the third to the number of solution points. In our
above example ``/soln/p3-tri`` has a length of 3231 since there are 3231
triangular elements in our mesh. The ``p3`` prefix indicates that each
of these triangles contains a third order solution polynomial which, in
turn, leads to 10 solution points. The locations of the solution points
can be determined in one of two ways. The first is to parse the
``/config`` dataset and the second is to inspect the ``pts`` attribute.
For ``/soln/p3-tri`` we find::

   ATTRIBUTE "pts" {
      DATATYPE  H5T_IEEE_F64LE
      DATASPACE  SIMPLE { ( 10, 2 ) / ( 10, 2 ) }
      DATA {
      (0,0): -0.333333, -0.333333,
      (1,0): -0.888872, 0.777744,
      (2,0): 0.777744, -0.888872,
      (3,0): -0.888872, -0.888872,
      (4,0): 0.268421, -0.408933,
      (5,0): -0.859489, -0.408933,
      (6,0): -0.408933, -0.859489,
      (7,0): 0.268421, -0.859489,
      (8,0): -0.859489, 0.268421,
      (9,0): -0.408933, 0.268421
      }
   }

The ``-parts`` suffixed array contains the MPI rank number that was
responsible for each element type.

Subset solutions
~~~~~~~~~~~~~~~~

It is permissible for solutions to be subset. If a particular element
type is subset then there will be an ``-idxs`` suffixed array in the
data group. This array will be of the same length as the data array and
will contain the numbers of the elements in the solution array. These
element numbers are guaranteed to be ascending.
