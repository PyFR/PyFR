.. highlight:: none

**********
Command Line Interface
**********

.. _cli:

More information on the commands and there inputs can be found by using
``--help``. Example::

     pyfr run --help

Major Commands
==============

There are four major commands of PyFR, these are:

1. ``pyfr import`` --- convert a `Gmsh
   <http:http://geuz.org/gmsh/>`_ .msh file into a PyFR .pyfrm file.

   Example::

        pyfr import mesh.msh mesh.pyfrm

2. ``pyfr partition`` --- partition an existing mesh and
   associated solution files.

   Example::

        pyfr partition 2 mesh.pyfrm solution.pyfrs .

3. ``pyfr run`` --- start a new PyFR simulation. Example::

        pyfr run mesh.pyfrm configuration.ini

4. ``pyfr restart`` --- restart a PyFR simulation from an existing
   solution file. Example::

        pyfr restart mesh.pyfrm solution.pyfrs

5. ``pyfr export`` --- convert a PyFR ``.pyfrs`` file into an unstructured
   VTK ``.vtu`` or ``.pvtu`` file. If a ``-k`` flag is provided with an integer
   argument then ``.pyfrs`` elements are converted to high-order VTK cells
   which are exported, where the order of the VTK cells is equal to the value
   of the integer argument.
   Example::

        pyfr export -k 4 mesh.pyfrm solution.pyfrs solution.vtu

   If a ``-d`` flag is provided with an integer argument then ``.pyfrs``
   elements are subdivided into linear VTK cells which are exported, where the
   number of sub-divisions is equal to the value of the integer argument.
   Example::

        pyfr export -d 4 mesh.pyfrm solution.pyfrs solution.vtu

   If no flags are provided then ``.pyfrs`` elements are converted to high-order
   VTK cells which are exported, where the order of the cells is equal to the
   order of the solution data in the ``.pyfrs`` file.

Plugin Commands
===============

Plugins are able to define tools that are accessible from the command line,
below are listed the the various options currently available.

