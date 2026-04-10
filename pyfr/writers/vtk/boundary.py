from collections import defaultdict, namedtuple

import numpy as np

from pyfr.cache import memoize
from pyfr.plugins.postproc.adapters import (BoundaryPostProcAdapter,
                                            GradBoundaryPostProcAdapter)
from pyfr.polys import get_polybasis
from pyfr.shapes import BaseShape
from pyfr.util import first, subclass_where
from pyfr.writers.vtk.base import BaseVTKWriter, interpolate_pts


FaceInfo = namedtuple('FaceInfo', 'etype fidx svpts norm')


def _search(a, v):
    idx = np.argsort(a)
    return idx[np.searchsorted(a, v, sorter=idx)]


class VTKBoundaryWriter(BaseVTKWriter):
    type = 'boundary'
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

                eoff = _search(smesh.eidxs[etype], eidxs[eoff])

            # Obtain the associated surface info
            for stype, *info in self._get_surface_info(etype, eoff, fidx):
                ecount[stype] += len(info[-1])
                self._surface_info[stype].append(info)

        self.einfo = list(ecount.items())

    @memoize
    def _get_shape(self, etype, cfg):
        nspts = len(self.mesh.spts[etype])
        return subclass_where(BaseShape, name=etype)(nspts, cfg)

    @memoize
    def _itype_opmats(self, etype, fidx, cfg):
        shape = self._get_shape(etype, cfg)

        # Get the information about our face
        itype, proj, norm = shape.faces[fidx]
        ishapecls = subclass_where(BaseShape, name=itype)

        # Obtain the visualisation points on this face
        svpts = np.array(ishapecls.std_ele(self.etypes_div[itype]))
        svpts = np.vstack(np.broadcast_arrays(*proj(*svpts.T))).T

        if self.ho_output:
            svpts = svpts[self._nodemaps[itype, len(svpts)]]

        mesh_op = shape.sbasis.nodal_basis_at(svpts)
        soln_op = shape.ubasis.nodal_basis_at(svpts)

        # Linear basis for P1 vertex data
        linspts = subclass_where(BaseShape, name=etype).std_ele(1)
        lbasis = get_polybasis(etype, 1, linspts)
        lin_op = lbasis.nodal_basis_at(svpts)

        finfo = FaceInfo(etype, fidx, svpts, norm)

        return itype, mesh_op, soln_op, lin_op, finfo

    def _get_surface_info(self, etype, eoffs, fidxs):
        info, idxs = {}, defaultdict(list)

        for e, f in zip(eoffs, fidxs):
            if f not in info:
                info[f] = self._itype_opmats(etype, f, self.cfg)

            idxs[f].append(e)

        return [(*info[f], idxs[f]) for f in info]

    def _make_adapter(self, face_vsoln, face_vpts, spts, finfo):
        if self._gradients:
            cls = GradBoundaryPostProcAdapter
        else:
            cls = BoundaryPostProcAdapter

        return cls(self, face_vsoln, face_vpts, spts, finfo)

    def _extra_point_shapes(self, key):
        if key in self._surface_info:
            etypes = [info[-2].etype for info in self._surface_info[key]]
        else:
            etypes = [key]

        shapes = set()
        for etype in etypes:
            nupts = self.soln.data[etype].shape[0]
            shapes.add((nupts,))
            shape = self._get_shape(etype, self.cfg)
            shapes.add((len(shape.linspts),))

        return shapes

    def _resolve_etype(self, key):
        if key is None:
            key = first(self._surface_info)

        if key in self._surface_info:
            key = first(self._surface_info[key])[-2].etype

        return key

    def _prepare_pts(self, itype):
        vspts, vsoln, curved = [], [], []
        cellf, pointf = defaultdict(list), defaultdict(list)

        pshapes = self._extra_point_shapes(itype)
        for mesh_op, soln_op, lin_op, finfo, idxs in self._surface_info[itype]:
            etype = finfo.etype
            spts = self.mesh.spts[etype][:, idxs]
            soln = self.soln.data[etype][..., idxs]
            soln = soln.swapaxes(0, 1).astype(self.dtype)

            # Pre-process the solution
            soln = self._pre_proc_fields(soln).swapaxes(0, 1)

            face_vpts = interpolate_pts(mesh_op, spts)
            face_vsoln = interpolate_pts(soln_op, soln)

            # Run postproc plugins
            if self.pp_plugins:
                adapter = self._make_adapter(face_vsoln, face_vpts, spts, finfo)
                for fname, arrs in self._run_postprocs(adapter).items():
                    pointf[fname].append(arrs)

            vspts.append(face_vpts)
            vsoln.append(face_vsoln)
            curved.append(self.mesh.spts_curved[etype][idxs])

            # Extract extra fields
            for fname, arr in self.soln.aux.get(etype, {}).items():
                data = arr[idxs]
                shape = data.shape[1:]
                if shape == (soln.shape[0],):
                    pointf[fname].append(
                        interpolate_pts(soln_op, data.swapaxes(0, 1))
                    )
                elif shape in pshapes:
                    pointf[fname].append(
                        interpolate_pts(lin_op, data.swapaxes(0, 1))
                    )
                else:
                    cellf[fname].append(data)

        # Concatenate extra fields
        cellf = {k: np.hstack(v) for k, v in cellf.items()}
        pointf = {k: np.hstack(v) for k, v in pointf.items()}

        return (np.hstack(vspts), np.dstack(vsoln),
                np.hstack(curved), cellf, pointf)
