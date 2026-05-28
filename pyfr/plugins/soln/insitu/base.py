import numpy as np

from pyfr.shapes import BaseShape, proj_pts
from pyfr.util import paren_depths, subclass_where
from pyfr.writers.vtk.clean import CleanToGrid


def con_psolns_pgrads(elementscls, scfg, csolns, cgrads):
    psolns = elementscls.con_to_pri(csolns, scfg)

    if cgrads is None:
        return psolns, None
    else:
        return psolns, elementscls.grad_con_to_pri(csolns, cgrads, scfg)


def face_shape_ops(etype, fidx, divisor, nspts, scfg):
    # Interp ops for sub-divided sample pts on face fidx of etype
    shapecls = subclass_where(BaseShape, name=etype)
    itype, proj, _ = shapecls.faces[fidx]
    ishapecls = subclass_where(BaseShape, name=itype)
    fsvpts = proj_pts(proj, ishapecls.std_ele(divisor))

    shape = shapecls(nspts, scfg)
    mesh_op = shape.sbasis.nodal_basis_at(fsvpts)
    soln_op = shape.ubasis.nodal_basis_at(fsvpts)

    return itype, mesh_op, soln_op, fsvpts


def split_components(expr):
    # Split on top-level commas; respect bracket/paren/brace nesting
    parts, buf = [], []
    for c, d in paren_depths(expr):
        if c == ',' and d == 0:
            parts.append(''.join(buf).strip())
            buf = []
        else:
            buf.append(c)

    parts.append(''.join(buf).strip())

    return parts


def build_cleaner(mesh, divisor, cnodemap, svptsmap):
    divmap = dict.fromkeys(cnodemap, divisor)
    shared = np.fromiter(mesh.shared_nodes.by_node, dtype=int)
    return CleanToGrid(cnodemap, divmap, svptsmap, shared)


def bp_key(k):
    return k.replace('_', '/').replace('-', '_')


class IntegratorAdapter:
    has_grads = True

    def __init__(self, intg, acfg, cfgsect):
        self.intg = intg
        self.mesh = intg.system.mesh
        self.scfg = intg.cfg
        self.acfg = acfg
        self.cfgsect = cfgsect
        self.dtype = intg.system.backend.fpdtype
        self.elementscls = intg.system.elementscls

    @property
    def tcurr(self):
        return self.intg.tcurr

    @property
    def cycle(self):
        return self.intg.nacptsteps

    @property
    def soln(self):
        return dict(zip(self.intg.system.ele_types, self.intg.soln))

    @property
    def grad_soln(self):
        return dict(zip(self.intg.system.ele_types, self.intg.grad_soln))

    def psolns_pgrads(self, csolns, cgrads):
        return con_psolns_pgrads(self.elementscls, self.scfg, csolns, cgrads)

    def soln_op_vpts(self, etype, divisor):
        eles = self.intg.system.ele_map[etype]
        shapecls = subclass_where(BaseShape, name=etype)
        shape = shapecls(eles.nspts, self.scfg)

        svpts = shape.std_ele(divisor)
        soln_op = shape.ubasis.nodal_basis_at(svpts).astype(self.dtype)

        return soln_op, eles.ploc_at_np(svpts)

    def face_soln_op_vpts(self, etype, fidx, divisor):
        nspts = self.intg.system.ele_map[etype].nspts
        ops = face_shape_ops(etype, fidx, divisor, nspts, self.scfg)
        itype, mop, sop, svpts = ops
        return itype, mop, sop.astype(self.dtype), svpts
