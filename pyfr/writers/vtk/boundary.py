from collections import defaultdict, namedtuple

import numpy as np

FaceInfo = namedtuple('FaceInfo',
                      ['etype', 'mesh_op', 'soln_op',
                       'jac_op', 'norm', 'idxs'])

from pyfr.cache import memoize
from pyfr.shapes import BaseShape
from pyfr.util import subclass_where
from pyfr.writers.vtk.base import BaseVTKWriter, interpolate_pts


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

        # Jacobian operator for normal computation
        jac_op = np.rollaxis(shape.sbasis.jac_nodal_basis_at(svpts), 2)

        return itype, mesh_op, soln_op, jac_op, norm

    def _get_surface_info(self, etype, eoffs, fidxs):
        info, idxs = {}, defaultdict(list)

        for e, f in zip(eoffs, fidxs):
            if f not in info:
                info[f] = self._itype_opmats(etype, f, self.cfg)

            idxs[f].append(e)

        return [(info[f][0],
                 FaceInfo(etype=etype, mesh_op=info[f][1],
                          soln_op=info[f][2], jac_op=info[f][3],
                          norm=info[f][4], idxs=idxs[f]))
                for f in info]

    def _prepare_pts(self, itype):
        vspts, vsoln, curved, part, vnorms = [], [], [], [], []
        needs_normals = any(pp.needs_normals for pp in self.pp_plugins)

        for fi in self._surface_info[itype]:
            spts = self.mesh.spts[fi.etype][:, fi.idxs]
            soln = self.soln[fi.etype][..., fi.idxs]
            soln = soln.swapaxes(0, 1).astype(self.dtype)

            # Pre-process the solution
            soln = self._pre_proc_fields(soln).swapaxes(0, 1)

            vspts.append(interpolate_pts(fi.mesh_op, spts))
            vsoln.append(interpolate_pts(fi.soln_op, soln))
            curved.append(self.mesh.spts_curved[fi.etype][fi.idxs])
            part.append(self.soln[f'{fi.etype}-parts'][fi.idxs])

            if needs_normals:
                vnorms.append(self._compute_normals(fi.jac_op, spts, fi.norm))

        vpts = np.hstack(vspts)
        vsoln = np.dstack(vsoln)
        normals = np.hstack(vnorms) if vnorms else None

        # Run postproc plugins
        vpts, vsoln = self._run_postprocs(vpts, vsoln, normals=normals)

        return vpts, vsoln, np.hstack(curved), np.hstack(part)

    def _compute_normals(self, jac_op, spts, ref_norm):
        ndims = self.ndims
        nspts, neles = spts.shape[0], spts.shape[1]

        # Compute Jacobian at vis points
        jac = jac_op.reshape(-1, nspts) @ spts.reshape(nspts, -1)
        nsvpts = jac.shape[0] // ndims
        jac = jac.reshape(nsvpts, ndims, ndims, neles)
        jac = jac.transpose(1, 2, 0, 3)  # (ndims, ndims, nsvpts, neles)

        # Compute smats
        smats = np.empty((ndims, nsvpts, ndims, neles))

        if ndims == 2:
            a, b = jac[0, 0], jac[1, 0]
            c, d = jac[0, 1], jac[1, 1]
            smats[0, :, 0], smats[0, :, 1] = d, -b
            smats[1, :, 0], smats[1, :, 1] = -c, a
        else:
            a, b, c = jac[0, 0], jac[1, 0], jac[2, 0]
            d, e, f = jac[0, 1], jac[1, 1], jac[2, 1]
            g, h, k = jac[0, 2], jac[1, 2], jac[2, 2]
            smats[0, :, 0] = e*k - f*h
            smats[0, :, 1] = f*g - d*k
            smats[0, :, 2] = d*h - e*g
            smats[1, :, 0] = c*h - b*k
            smats[1, :, 1] = a*k - c*g
            smats[1, :, 2] = b*g - a*h
            smats[2, :, 0] = b*f - c*e
            smats[2, :, 1] = c*d - a*f
            smats[2, :, 2] = a*e - b*d

        # Physical normal: pnorm = S^T . n_ref
        ref_norm = np.array(ref_norm, dtype=float)
        pnorm = np.einsum('ijkl,k->ijl', smats, ref_norm)

        # Normalize
        mag = np.sqrt(np.einsum('ijk,ijk->jk', pnorm, pnorm))
        pnorm /= mag[np.newaxis]

        return pnorm  # (ndims, nsvpts, neles)
