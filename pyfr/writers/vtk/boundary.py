from collections import defaultdict, namedtuple

import numpy as np

FaceInfo = namedtuple('FaceInfo',
                      ['etype', 'fidx', 'mesh_op', 'soln_op',
                       'svpts', 'norm', 'idxs'])

from pyfr.cache import memoize
from pyfr.shapes import BaseShape
from pyfr.util import subclass_where
from pyfr.writers.vtk.base import (BaseVTKWriter, _BoundaryPostProcAdapter,
                                   interpolate_pts)


def _search(a, v):
    idx = np.argsort(a)
    return idx[np.searchsorted(a, v, sorter=idx)]


class VTKBoundaryWriter(BaseVTKWriter):
    type = 'boundary'
    output_curved = True
    output_partition = True

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
            for stype, finfo in self._get_surface_info(etype, eoff, fidx):
                ecount[stype] += len(finfo.idxs)
                self._surface_info[stype].append(finfo)

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

        return itype, mesh_op, soln_op, svpts, norm

    def _get_surface_info(self, etype, eoffs, fidxs):
        info, idxs = {}, defaultdict(list)

        for e, f in zip(eoffs, fidxs):
            if f not in info:
                info[f] = self._itype_opmats(etype, f, self.cfg)

            idxs[f].append(e)

        return [(info[f][0],
                 FaceInfo(etype=etype, fidx=f, mesh_op=info[f][1],
                          soln_op=info[f][2], svpts=info[f][3],
                          norm=info[f][4], idxs=idxs[f]))
                for f in info]

    def _prepare_pts(self, itype):
        vspts, vsoln, curved, part = [], [], [], []

        for fi in self._surface_info[itype]:
            spts = self.mesh.spts[fi.etype][:, fi.idxs]
            soln = self.soln[fi.etype][..., fi.idxs]
            soln = soln.swapaxes(0, 1).astype(self.dtype)

            # Pre-process the solution
            soln = self._pre_proc_fields(soln).swapaxes(0, 1)

            face_vpts = interpolate_pts(fi.mesh_op, spts)
            face_vsoln = interpolate_pts(fi.soln_op, soln)

            # Run postproc plugins per-batch with full shape context
            if self.pp_plugins:
                adapter = _BoundaryPostProcAdapter(
                    self, face_vpts, face_vsoln,
                    self._get_shape(fi.etype, self.cfg), spts,
                    fi.fidx, fi.svpts, fi.norm
                )
                face_vsoln = self._run_postprocs(adapter, face_vsoln)

            vspts.append(face_vpts)
            vsoln.append(face_vsoln)
            curved.append(self.mesh.spts_curved[fi.etype][fi.idxs])
            part.append(self.soln[f'{fi.etype}-parts'][fi.idxs])

        return (np.hstack(vspts), np.dstack(vsoln),
                np.hstack(curved), np.hstack(part))
