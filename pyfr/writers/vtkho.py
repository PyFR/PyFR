# -*- coding: utf-8 -*-

import numpy as np

from pyfr.shapes import BaseShape
from pyfr.util import subclass_where
from pyfr.writers.nodemaps import VTKHONodeMaps
from pyfr.writers.vtk import VTKWriter


class VTKHOWriter(VTKWriter):
    name = 'vtk-ho'
    extn = ['']

    vtk_types = dict(tri=69, quad=70, tet=71, pri=73, hex=72)

    # With VTKFile version >= 2.1, VTK>=9 supposes
    # that hex connectivity corresponds to
    # VTK9 mapping
    _vtkfile_version = '2.1'

    def _get_npts_ncells_nnodes(self, sk):
        etype, neles = self.soln_inf[sk][0], self.soln_inf[sk][1][2]

        if etype == 'pyr':
            # No Lagrange pyr cells in VTK
            # Therefore, rely on the subdivision mechanism
            # of the vtk writer
            return super()._get_npts_ncells_nnodes(sk)

        # Get the shape and sub division classes
        shapecls = subclass_where(BaseShape, name=etype)

        # Number of vis points
        # which coincides with the number of
        # nodes of the vtkLagrange* correspondent
        # objects
        npts = shapecls.nspts_from_order(self.divisor + 1)*neles

        return npts, neles, npts

    def _write_data(self, vtuf, mk, sk):
        etype = self.mesh_inf[mk][0]

        if etype == 'pyr':
            # No Lagrange pyr cells in VTK
            # Therefore, rely on the subdivision mechanism
            # of the vtk writer
            return super()._write_data(vtuf, mk, sk)

        mesh = self.mesh[mk].astype(self.dtype)
        soln = self.soln[sk].swapaxes(0, 1).astype(self.dtype)

        # Handle the case of partial solution files
        if soln.shape[2] != mesh.shape[1]:
            skpre, skpost = sk.rsplit('_', 1)

            mesh = mesh[:, self.soln[f'{skpre}_idxs_{skpost}'], :]

        # Dimensions
        nspts, neles = mesh.shape[:2]

        # Points inside of a standard element
        svpts = self._get_std_ele(etype, nspts)
        nsvpts = len(svpts)

        # Transform PyFR to VTK9 points
        # Read nodemaps.py for more information on
        # why we chose VTK9 over VTK8 maps
        svpts = np.array(svpts)[VTKHONodeMaps.from_pyfr[etype, nsvpts]]

        # Generate the operator matrices
        mesh_vtu_op = self._get_mesh_op(etype, nspts, svpts)
        soln_vtu_op = self._get_soln_op(etype, nspts, svpts)

        # Calculate node locations of VTU elements
        vpts = mesh_vtu_op @ mesh.reshape(nspts, -1)
        vpts = vpts.reshape(nsvpts, -1, self.ndims)

        # Pre-process the solution
        soln = self._pre_proc_fields(etype, mesh, soln).swapaxes(0, 1)

        # Interpolate the solution to the vis points
        vsoln = soln_vtu_op @ soln.reshape(len(soln), -1)
        vsoln = vsoln.reshape(nsvpts, -1, neles).swapaxes(0, 1)

        # Append dummy z dimension for points in 2D
        if self.ndims == 2:
            vpts = np.pad(vpts, [(0, 0), (0, 0), (0, 1)], 'constant')

        # Write element node locations to file
        self._write_darray(vpts.swapaxes(0, 1), vtuf, self.dtype)

        # Prepare VTU cell arrays
        vtu_con = np.tile(np.arange(0, nsvpts), (neles, 1))
        vtu_con += (np.arange(neles)*nsvpts)[:, None]

        # Generate offset into the connectivity array
        vtu_off = np.tile(nsvpts, (neles, 1))
        vtu_off += (np.arange(neles)*nsvpts)[:, None]

        # Tile VTU cell type numbers
        vtu_typ = np.tile(self.vtk_types[etype], neles)

        # Write VTU node connectivity, connectivity offsets and cell types
        self._write_darray(vtu_con, vtuf, np.int32)
        self._write_darray(vtu_off, vtuf, np.int32)
        self._write_darray(vtu_typ, vtuf, np.uint8)

        # Process and write out the various fields
        for arr in self._post_proc_fields(vsoln):
            self._write_darray(arr.T, vtuf, self.dtype)
