from collections import defaultdict

import numpy as np

from pyfr.cache import memoize
from pyfr.fields.geometry import BoundaryGeometry
from pyfr.nputil import search_unsorted
from pyfr.polys import get_polybasis
from pyfr.shapes import BaseShape, proj_pts
from pyfr.util import subclass_where
from pyfr.writers.vtk.base import BaseVTKWriter, interpolate_pts
from pyfr.writers.vtk.shapes import get_vtk_shape


class VTKBoundaryWriter(BaseVTKWriter):
    type = 'boundary'
    dimensions = '2|3'
    output_curved = True

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
        cidxs = [smesh.codec.index(f'bc/{b}') for b in self.boundaries]

        for etype, einfo in self.reader.eles.items():
            # See which of our faces are on the selected boundaries
            mask = np.isin(einfo['faces']['cidx'], cidxs)
            eoff, fidx = mask.nonzero()

            # Handle the case where the solution is subset
            if smesh.subset and eoff.any():
                # Ensure this element type is present in the subset
                if etype not in smesh.eidxs:
                    raise ValueError('Output boundaries not present in subset '
                                     'solution')

                # Ensure all of the required element numbers are present
                eidxs = rmesh.eidxs[etype]
                beidx = eidxs[mask.any(axis=1)]
                if not np.isin(beidx, smesh.eidxs[etype]).all():
                    raise ValueError('Output boundaries not present in subset '
                                     'solution')

                eoff = search_unsorted(smesh.eidxs[etype], eidxs[eoff])

            # Obtain the associated surface info
            for stype, *info in self._get_surface_info(etype, eoff, fidx):
                ecount[stype] += len(info[-1])
                self._surface_info[stype].append(info)

        self.einfo = list(ecount.items())

    @memoize
    def _itype_opmats(self, etype, fidx, cfg):
        shape = self._get_shape(etype, cfg)

        # Get the information about our face
        itype, proj, _ = shape.faces[fidx]

        # Obtain the visualisation points on this face
        svpts = proj_pts(proj, self._svpts(itype))

        mesh_op = shape.sbasis.nodal_basis_at(svpts)
        soln_op = shape.ubasis.nodal_basis_at(svpts)

        # Linear basis for P1 vertex data
        linspts = subclass_where(BaseShape, name=etype).std_ele(1)
        lbasis = get_polybasis(etype, 1, linspts)
        lin_op = lbasis.nodal_basis_at(svpts)

        return itype, mesh_op, soln_op, lin_op, etype, fidx, svpts

    @memoize
    def _svpts(self, itype):
        ishapecls = subclass_where(BaseShape, name=itype)
        svpts = ishapecls.std_ele(self.etypes_div[itype])
        if self.ho_output:
            vshape = get_vtk_shape(itype, self.etypes_div[itype])
            svpts = svpts[vshape.nodemaps[len(svpts)]]
        return svpts

    def _output_topology(self):
        svpts = {itype: self._svpts(itype) for itype in self._surface_info}

        cnodes = {}
        for itype, groups in self._surface_info.items():
            pieces = []
            for *_, etype, fidx, _, idxs in groups:
                shapecls = subclass_where(BaseShape, name=etype)
                spts_nodes = self.mesh.spts_nodes[etype]
                cidxs = shapecls.face_corner_pts_idxs(fidx,
                                                      spts_nodes.shape[1])
                pieces.append(spts_nodes[np.ix_(idxs, cidxs)])

            cnodes[itype] = np.concatenate(pieces)

        return cnodes, svpts

    def _get_surface_info(self, etype, eoffs, fidxs):
        info, idxs = {}, defaultdict(list)

        for e, f in zip(eoffs, fidxs):
            if f not in info:
                info[f] = self._itype_opmats(etype, f, self.cfg)

            idxs[f].append(e)

        return [(*info[f], idxs[f]) for f in info]

    def _itype_point_shapes(self, itype):
        shapes = set()
        for *_, etype, _, _, _ in self._surface_info[itype]:
            shapes.update(self._extra_point_shapes(etype))
        return shapes

    def _prepare_pts(self, itype):
        vspts, vsoln, curved = [], [], []
        cellf, pointf = defaultdict(list), defaultdict(list)

        pshapes = self._itype_point_shapes(itype)
        for *ops, etype, fidx, svpts, idxs in self._surface_info[itype]:
            mesh_op, soln_op, lin_op = ops
            spts = self.mesh.spts[etype][:, idxs]
            soln = self.soln.data[etype][..., idxs]
            soln = soln.swapaxes(0, 1).astype(self.dtype)

            # Pre-process the solution
            soln = self._pre_proc_fields(soln).swapaxes(0, 1)

            face_vpts = interpolate_pts(mesh_op, spts)
            face_vsoln = interpolate_pts(soln_op, soln)

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
                    interpolate_pts(op, np.moveaxis(data, 0, 1))
                )

            samples = face_vsoln.transpose(1, 0, 2)

            # Evaluate any registry-backed fields
            if self._reg_fields:
                geom = None
                if self._registry.needs_geom(self._reg_fields):
                    bg = BoundaryGeometry(self.soln.config, spts, etype,
                                          fidx, svpts)
                    geom = bg.symbols(self.ndims)

                for fname, arr in self._registry.evaluate(
                        self._reg_fields, samples, geom=geom).items():
                    pointf[fname].append(arr.astype(self.dtype))

        # Concatenate extra fields
        cellf = {k: np.hstack(v) for k, v in cellf.items()}
        pointf = {k: np.hstack(v) for k, v in pointf.items()}

        return (np.hstack(vspts), np.dstack(vsoln), np.hstack(curved), cellf,
                pointf)
