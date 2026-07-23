import numpy as np

from pyfr.cache import memoize
from pyfr.polys import get_polybasis
from pyfr.shapes import BaseShape
from pyfr.util import subclass_where
from pyfr.writers.vtk.base import BaseVTKWriter, interpolate_pts
from pyfr.writers.vtk.shapes import get_vtk_shape


class VTKVolumeWriter(BaseVTKWriter):
    type = 'volume'
    dimensions = '2|3'
    output_curved = True

    def _load_soln(self, *args, **kwargs):
        super()._load_soln(*args, **kwargs)

        self.einfo = [(etype, self.soln.data[etype].shape[2])
                      for etype in self.mesh.eidxs]

    def _output_topology(self):
        cnodes, svpts = {}, {}
        for etype, _ in self.einfo:
            shapecls = subclass_where(BaseShape, name=etype)
            spts_nodes = self.mesh.spts_nodes[etype]
            cidxs = shapecls.corner_pts_idxs(spts_nodes.shape[1])
            cnodes[etype] = spts_nodes[:, cidxs]
            svpts[etype] = self._svpts(etype)

        return cnodes, svpts

    @memoize
    def _svpts(self, etype):
        shapecls = subclass_where(BaseShape, name=etype)
        div = self.etypes_div[etype]
        svpts = shapecls.std_ele(div)

        # For high-order output permute the nodes to match the VTK ordering
        if etype != 'pyr' and self.ho_output:
            svpts = svpts[get_vtk_shape(etype, div).nodemaps[len(svpts)]]

        return svpts

    @memoize
    def _opmats(self, etype, cfg):
        # Shape
        shapecls = subclass_where(BaseShape, name=etype)

        # Sub divison points inside of a standard element
        svpts = self._svpts(etype)

        # Basis
        basis = shapecls(len(self.mesh.spts[etype]), cfg)

        mesh_op = basis.sbasis.nodal_basis_at(svpts)
        soln_op = basis.ubasis.nodal_basis_at(svpts)

        # Linear basis for vertex data
        linspts = shapecls.std_ele(1)
        lbasis = get_polybasis(etype, 1, linspts)
        lin_op = lbasis.nodal_basis_at(svpts)

        return mesh_op, soln_op, lin_op

    def _prepare_pts(self, etype):
        spts = self.mesh.spts[etype].astype(self.dtype)
        soln = self.soln.data[etype].swapaxes(0, 1).astype(self.dtype)
        curved = self.mesh.spts_curved[etype]

        # Initialise extra field dicts
        cellf, pointf = {}, {}

        # Generate the interpolation operator matrices
        mesh_vtu_op, soln_vtu_op, lin_vtu_op = self._opmats(etype, self.cfg)

        # Calculate node locations of VTU elements
        vpts = interpolate_pts(mesh_vtu_op, spts)

        # Pre-process the solution at upts
        soln = self._pre_proc_fields(soln).swapaxes(0, 1)

        # Interpolate the solution to the vis points
        vsoln = interpolate_pts(soln_vtu_op, soln)

        samples = vsoln.transpose(1, 0, 2)

        # Evaluate any registry-backed fields
        if self._reg_fields:
            for fname, arr in self._registry.evaluate(self._reg_fields,
                                                      samples).items():
                pointf[fname] = arr.astype(self.dtype)

        # Append dummy z dimension for points in 2D (post-pp)
        if self.ndims == 2:
            vpts = np.pad(vpts, [(0, 0), (0, 0), (0, 1)], 'constant')

        # Extract extra fields
        nupts = soln.shape[0]
        pshapes = self._extra_point_shapes(etype)
        for fname, data in self.soln.aux.get(etype, {}).items():
            shape = data.shape[1:]

            if shape in pshapes:
                pshape = shape
            elif shape[:-1] in pshapes:
                pshape = shape[:-1]
            else:
                cellf[fname] = data
                continue

            op = soln_vtu_op if pshape == (nupts,) else lin_vtu_op
            pointf[fname] = interpolate_pts(op, np.moveaxis(data, 0, 1))

        return vpts, vsoln, curved, cellf, pointf
