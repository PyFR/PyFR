from collections import namedtuple

import numpy as np

from pyfr.mpiutil import autofree, get_comm_rank_root
from pyfr.shapes import BaseShape
from pyfr.util import subclasses


Metrics = namedtuple('Metrics',
                     'djac h aspect scaled_jac jvar vol char_len ploc')


def _scaled_jac(ele):
    sj = None

    for name in ('upts', 'fpts'):
        jac = ele.jac_at_np(name)

        # Normalise the determinant by the tangent vector norms
        tnorm = np.prod(np.linalg.norm(jac, axis=-1), axis=-1)
        with np.errstate(divide='ignore', invalid='ignore'):
            s = np.where(tnorm > 0, np.linalg.det(jac) / tnorm, 0)

        # Keep the worst value seen at any of the solver's points
        s = np.min(s, axis=0)
        sj = s if sj is None else np.minimum(sj, s)

    return sj


def element_metrics(ele):
    # Jacobian determinant at solution points
    djac = np.linalg.det(ele.jac_at_np('upts'))
    with np.errstate(divide='ignore', invalid='ignore'):
        rcpdjac = 1.0 / djac

    # Metric terms at solution points
    smats = ele.smat_at_np('upts')

    # J^{-1} scaled by 1/det(J)
    jinv = smats * rcpdjac[None, :, None, :]

    # Mesh scale: h_i = 2 / ||J^{-1}_i||_2 per reference direction
    h_per_dir = 2.0 / np.sqrt(np.sum(jinv**2, axis=2))
    h_min = np.min(h_per_dir, axis=0)
    h_max = np.max(h_per_dir, axis=0)

    # Aspect ratio
    aspect = h_max / h_min

    # Scaled Jacobian at the solution and flux points
    scaled_jac = _scaled_jac(ele)

    # Variation of the Jacobian determinant within each element
    adjac = np.abs(djac)
    with np.errstate(divide='ignore', invalid='ignore'):
        jvar = np.min(adjac, axis=0) / np.max(adjac, axis=0)

    # Element volume from the solution point quadrature
    wts = ele.basis.ubasis.invvdm[:, 0]
    vol = wts.sum() * (wts @ djac)

    # Volume-based characteristic length
    char_len = np.abs(np.mean(djac, axis=0))**(1.0 / ele.ndims)

    # Physical locations
    ploc = ele.ploc_at_np('upts')

    return Metrics(djac, h_min, aspect, scaled_jac, jvar, vol, char_len, ploc)


def neighbour_size_ratio(mesh, char_len):
    comm, _, _ = get_comm_rank_root()

    nsr = {et: np.ones(len(cl)) for et, cl in char_len.items()}

    def process(con, lcl, rcl):
        face_r = np.maximum(lcl, rcl) / np.minimum(lcl, rcl)
        for et, _, ei, mask in con.foreach():
            np.maximum.at(nsr[et], ei, face_r[mask])

    # Internal faces
    if mesh.con:
        for lhs, rhs in [mesh.con, mesh.con[::-1]]:
            process(lhs, lhs.map_eles(char_len), rhs.map_eles(char_len))

    # MPI faces
    nbrs = sorted(mesh.con_p)
    ncomm = autofree(comm.Create_dist_graph_adjacent(nbrs, nbrs))
    send = [con.map_eles(char_len) for con in mesh.con_p.values()]
    recv = ncomm.neighbor_alltoall(send)
    for con, rcl in zip(mesh.con_p.values(), recv):
        process(con, con.map_eles(char_len), rcl)

    return nsr


class MeshQuality:
    # Per-element quality fields, in output order
    names = ('scaled-jacobian', 'jacobian-variation', 'element-volume',
             'mesh-scale', 'aspect-ratio', 'size-ratio')

    def __init__(self, mesh, cfg):
        from pyfr.plugins.common import get_elementscls

        elementscls = get_elementscls(cfg)
        basismap = {b.name: b for b in subclasses(BaseShape, just_leaf=True)}

        self.metrics = {}
        for etype, spts in mesh.spts.items():
            ele = elementscls(basismap[etype], spts, cfg)
            self.metrics[etype] = element_metrics(ele)

        char_len = {et: m.char_len for et, m in self.metrics.items()}
        self.nsr = neighbour_size_ratio(mesh, char_len)

    def cell_fields(self):
        # Reduce each metric to the value per element the report shows
        fields = {}
        for etype, m in self.metrics.items():
            vals = (m.scaled_jac, m.jvar, m.vol, np.min(m.h, axis=0),
                    np.max(m.aspect, axis=0), self.nsr[etype])
            fields[etype] = dict(zip(self.names, vals))

        return fields
