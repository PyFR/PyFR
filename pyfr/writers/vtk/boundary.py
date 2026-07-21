from collections import defaultdict

import numpy as np

from pyfr.cache import memoize
from pyfr.plugins.postproc.adapters import FaceInfo
from pyfr.polys import get_polybasis
from pyfr.shapes import BaseShape, interp_pts, proj_pts
from pyfr.subdiv import get_subdiv
from pyfr.util import subclass_where
from pyfr.writers.vtk.base import BaseVTKWriter


class VTKBoundaryWriter(BaseVTKWriter):
    type = 'boundary'
    dimensions = '2|3'
    output_curved = True
    needs_con = True

    def __init__(self, meshf, boundaries, **kwargs):
        super().__init__(meshf, **kwargs)

        self.boundaries = boundaries

        if self.ndims != 3:
            raise RuntimeError('Boundary export only supported for 3D grids')

    def _load_soln(self, *args, **kwargs):
        super()._load_soln(*args, **kwargs)

        ecount = defaultdict(int)
        self._surface_info = defaultdict(list)

        rmesh, smesh = self.reader.mesh, self.mesh
        for b in self.boundaries:
            if f'bc/{b}' not in smesh.codec:
                raise ValueError(f'Unknown boundary {b!r}')

            # Subset solutions must retain every local boundary face
            pcon, con = rmesh.bcon.get(b), smesh.bcon.get(b)
            if smesh.subset and len(con or []) != len(pcon or []):
                raise ValueError('Output boundaries not present in subset '
                                 'solution')

            for etype, fidx, eoff in (con.items() if con else []):
                itype, *info = self._itype_opmats(etype, fidx, self.cfg)
                ecount[itype] += len(eoff)
                self._surface_info[itype].append((*info, eoff))

        self.einfo = list(ecount.items())

    @memoize
    def _itype_opmats(self, etype, fidx, cfg):
        shape = self._get_shape(etype, cfg)

        # Get the information about our face
        itype, proj, norm = shape.faces[fidx]

        # Obtain the visualisation points on this face
        svpts = proj_pts(proj, self._svpts(itype))

        mesh_op = shape.sbasis.nodal_basis_at(svpts)
        soln_op = shape.ubasis.nodal_basis_at(svpts)

        # Linear basis for P1 vertex data
        linspts = subclass_where(BaseShape, name=etype).std_ele(1)
        lbasis = get_polybasis(etype, 1, linspts)
        lin_op = lbasis.nodal_basis_at(svpts)

        finfo = FaceInfo(etype, fidx, svpts, norm)

        return itype, mesh_op, soln_op, lin_op, finfo

    @memoize
    def _svpts(self, itype):
        ishapecls = subclass_where(BaseShape, name=itype)
        svpts = ishapecls.std_ele(self.etypes_div[itype])
        if self.ho_output:
            sdiv = get_subdiv(itype, self.etypes_div[itype])
            svpts = svpts[sdiv.vtk_nodemaps[len(svpts)]]
        return svpts

    def _output_topology(self):
        svpts = {itype: self._svpts(itype) for itype in self._surface_info}

        cnodes = {}
        for itype, groups in self._surface_info.items():
            pieces = []
            for *_, finfo, idxs in groups:
                shapecls = subclass_where(BaseShape, name=finfo.etype)
                spts_nodes = self.mesh.spts_nodes[finfo.etype]
                cidxs = shapecls.face_corner_pts_idxs(finfo.fidx,
                                                      spts_nodes.shape[1])
                pieces.append(spts_nodes[np.ix_(idxs, cidxs)])

            cnodes[itype] = np.concatenate(pieces)

        return cnodes, svpts

    def _itype_point_shapes(self, itype):
        shapes = set()
        for *_, finfo, _ in self._surface_info[itype]:
            shapes.update(self._extra_point_shapes(finfo.etype))
        return shapes

    def _prepare_pts(self, itype):
        vspts, vsoln, curved = [], [], []
        cellf, pointf = defaultdict(list), defaultdict(list)

        pshapes = self._itype_point_shapes(itype)
        for mesh_op, soln_op, lin_op, finfo, idxs in self._surface_info[itype]:
            etype = finfo.etype
            spts = self.mesh.spts[etype][:, idxs]
            soln = self.soln.data[etype][..., idxs]
            soln = soln.swapaxes(0, 1).astype(self.dtype)

            # Pre-process the solution
            soln = self._pre_proc_fields(soln).swapaxes(0, 1)

            face_vpts = interp_pts(mesh_op, spts)
            face_vsoln = interp_pts(soln_op, soln)

            vspts.append(face_vpts)
            vsoln.append(face_vsoln)
            curved.append(self.mesh.spts_curved[etype][idxs])

            # Extract aux fields
            nupts = soln.shape[0]
            for fname, arr in self.soln.aux.get(etype, {}).items():
                data = arr[idxs]
                shape = data.shape[1:]

                if shape in pshapes:
                    pshape = shape
                elif shape[:-1] in pshapes:
                    pshape = shape[:-1]
                else:
                    cellf[fname].append(data)
                    continue

                op = soln_op if pshape == (nupts,) else lin_op
                pointf[fname].append(
                    interp_pts(op, np.moveaxis(data, 0, 1))
                )

            soln_t = face_vsoln.transpose(1, 0, 2)
            ploc = face_vpts.transpose(2, 0, 1)

            fields = self._postproc(soln_t, ploc, self.elementscls, spts, finfo)
            for fname, arr in fields.items():
                pointf[fname].append(arr)

        # Concatenate extra fields
        cellf = {k: np.hstack(v) for k, v in cellf.items()}
        pointf = {k: np.hstack(v) for k, v in pointf.items()}

        return (np.hstack(vspts), np.dstack(vsoln), np.hstack(curved), cellf,
                pointf)
