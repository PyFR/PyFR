from collections import namedtuple
from functools import partial

import numpy as np

from pyfr.mpiutil import (DistributedDirectory, get_comm_rank_root, mpi,
                          scal_coll)
from pyfr.polys import get_polybasis
from pyfr.shapes import BaseShape, interp_pts
from pyfr.subdiv import get_subdiv
from pyfr.util import first, subclass_where, subclasses


KindGlobal = namedtuple('KindGlobal',
                        'keys defidx nint emap count reducer')


def _unique_rows(keys):
    n, k = keys.shape
    if k == 1:
        uniq, inv = np.unique(keys.ravel(), return_inverse=True)
        return uniq.reshape(-1, 1), inv

    # lexsort + first-of-group is faster than struct-view for k>1
    order = np.lexsort(keys.T)
    srt = keys[order]
    fmask = np.empty(n, dtype=bool)
    fmask[:1] = True
    fmask[1:] = (srt[1:] != srt[:-1]).any(axis=1)
    inv = np.empty(n, dtype=int)
    inv[order] = np.cumsum(fmask) - 1
    return srt[fmask], inv


class Reducer:
    # FxHash multiplicative-XOR constants (golden ratio and sign-bit mask)
    FX_SEED = np.uint64(0x9E3779B97F4A7C15)
    FX_MASK = np.uint64(0x7FFFFFFFFFFFFFFF)

    def __init__(self, comm, fkeys):
        self.dd = DistributedDirectory(comm, self._hash_rows(fkeys))
        self.uniq, self.inv = _unique_rows(self.dd.scatter(fkeys))

    def __call__(self, values):
        rv = self.dd.scatter(values)
        gv = np.zeros((len(self.uniq), *values.shape[1:]), dtype=values.dtype)
        np.add.at(gv, self.inv, rv)
        return self.dd.gather(gv[self.inv])

    @classmethod
    def _hash_rows(cls, keys):
        h = np.zeros(len(keys), dtype=np.uint64)
        for col in keys.view(np.uint64).T:
            h = (h*cls.FX_SEED) ^ col

        return h & cls.FX_MASK


class CleanToGrid:
    @staticmethod
    def _class_rows(classes, nint):
        return (classes[:, None]*nint + np.arange(nint)).ravel()

    def __init__(self, cnodemap, divmap, nsvptsmap, shared):
        self.comm, _, _ = get_comm_rank_root()
        self.cnodemap = cnodemap

        topos, edef, ktypes = {}, {}, set()
        for etype, cnodes in cnodemap.items():
            sdiv = get_subdiv(etype, divmap[etype])
            topos[etype] = sdiv.topology(nsvptsmap[etype])
            edef[etype] = np.isin(cnodes, shared).any(axis=1)
            ktypes.update(topos[etype].kinds)

        # Reduce to obtain the global set of entities
        ktypes = set().union(*self.comm.allgather(ktypes))

        # Per-kind cross-etype dedup -> self.kinds (with emap populated)
        self.kinds = {kind: self._init_kind(kind, topos, cnodemap, edef)
                      for kind in sorted(ktypes)}

        self.layouts = {etype: self._build_layout(etype, topos[etype])
                        for etype in cnodemap}

    def _init_kind(self, kind, topos, cnodemap, edef):
        evdofs, klist, dlist, nint, ncols = {}, [], [], 0, 0
        for etype, cn in cnodemap.items():
            if kind not in topos[etype].kinds:
                continue

            keys, fsrc, cpos, nint = topos[etype].canonical_vdofs(kind, cn)
            evdofs[etype] = (keys, fsrc, cpos)
            klist.append(keys)
            dlist.append(np.repeat(edef[etype], len(keys) // len(cn)))
            ncols = keys.shape[1]

        # Agree on nint and key width across ranks
        nint = scal_coll(self.comm.Allreduce, nint, op=mpi.MAX)
        ncols = scal_coll(self.comm.Allreduce, ncols, op=mpi.MAX)

        # Deduplicate canonical keys across all contributing etypes
        akeys = np.concatenate(klist or [np.empty((0, ncols), dtype=int)])
        adefs = np.concatenate(dlist or [np.empty(0, dtype=bool)])

        uniq, inv = _unique_rows(akeys)
        defidx = np.unique(inv[adefs])

        # Map each etype's vdofs to their global class indices
        emap, off = {}, 0
        for etype, (keys, fsrc, cpos) in evdofs.items():
            cls = inv[off:off + len(keys)]
            emap[etype] = (np.unique(cls), fsrc, cls*nint + cpos)
            off += len(keys)

        # Determine counts
        count = np.bincount(inv, minlength=len(uniq)) // max(nint, 1)
        reducer = Reducer(self.comm, uniq[defidx])
        count[defidx] = reducer(count[defidx])

        return KindGlobal(uniq, defidx, nint, emap, count, reducer)

    def _build_layout(self, etype, topo):
        neles, npts = len(self.cnodemap[etype]), topo.npts
        remap = np.empty((neles, npts), dtype=int)
        kept, off = [], 0

        # Merge coincident vdofs into unique kept positions
        for g in self.kinds.values():
            if (kp := g.emap.get(etype)) is None:
                continue

            # Local vdof positions within this etype's classes
            classes, fsrc, fdst = kp
            cls, cpos = divmod(fdst, g.nint)
            lcls = np.searchsorted(classes, cls)
            within = lcls*g.nint + cpos

            # First-occurrence source indices for each kept vdof
            ukept, fidx = np.unique(within, return_index=True)
            ksrc = np.empty(len(classes)*g.nint, dtype=int)
            ksrc[ukept] = fsrc[fidx]

            # Update the remap table and accumulate kept sources
            remap[divmod(fsrc, npts)] = off + within
            kept.append(ksrc)
            off += len(classes)*g.nint

        # Body vdofs: unique by construction (sequential kept ids)
        bidxs = topo.body_idxs
        bpos = ((np.arange(neles)*npts)[:, None] + bidxs).ravel()
        remap[:, bidxs] = off + np.arange(bpos.size).reshape(neles, -1)
        kept.append(bpos)

        return remap, np.concatenate(kept), bpos

    def select(self, etype, values):
        _, kept, _ = self.layouts[etype]
        flat = values.swapaxes(0, 1).reshape(-1, *values.shape[2:])
        return flat[kept]

    def average(self, fields, ncomp, dtype):
        # Per-kind: accumulate sums -> exchange deficient sums -> divide by
        # precomputed topological count
        vdofvals = {etype: vals.swapaxes(0, 1).reshape(-1, ncomp)
                    for etype, vals in fields.items()}

        out = {etype: [] for etype in fields}
        for g in self.kinds.values():
            psum = np.zeros((len(g.keys)*g.nint, ncomp), dtype=dtype)

            # Non-finite values from guarded ratios propagate by design
            with np.errstate(invalid='ignore'):
                for etype, (_, fsrc, fdst) in g.emap.items():
                    np.add.at(psum, fdst, vdofvals[etype][fsrc])

                rows = self._class_rows(g.defidx, g.nint)
                vals = psum[rows].reshape(-1, g.nint, ncomp)
                psum[rows] = g.reducer(vals).reshape(-1, ncomp)

                psum /= np.repeat(g.count, g.nint)[:, None]

            for etype, (classes, _, _) in g.emap.items():
                # When this etype's classes cover every global class for
                # this kind, the fancy-index slice is the whole array
                if len(classes) == len(g.keys):
                    out[etype].append(psum)
                else:
                    rows = self._class_rows(classes, g.nint)
                    out[etype].append(psum[rows])

        # Append body slice and concat per-etype
        for etype in fields:
            _, _, body = self.layouts[etype]
            out[etype].append(vdofvals[etype][body])
            out[etype] = np.concatenate(out[etype])

        return out


def con_block_to_pri(elementscls, cfg, ndims, block, *, grads=False,
                     resid=False):
    # Fields-first conservative block (+grads, +resid) to primitives
    nvars = len(elementscls.convars(ndims, cfg))
    fields = elementscls.con_to_pri(block[:nvars], cfg)

    # Solution gradients convert via the chain rule
    if grads:
        ng = nvars*(1 + ndims)
        dcon = block[nvars:ng].reshape(nvars, ndims, *block.shape[1:])
        dpri = elementscls.diff_con_to_pri(block[:nvars], dcon, cfg)

        fields += [f for gf in dpri for f in gf]

    # Residuals convert to primitive rates
    if resid:
        fields += elementscls.diff_con_to_pri(block[:nvars], block[-nvars:],
                                              cfg)

    return np.array(fields)


class FieldRecovery:
    def __init__(self, mesh, elementscls, cfg):
        self.order = p = cfg.getint('solver', 'order')
        self.mesh, self.cfg = mesh, cfg
        self.elementscls, self.ndims = elementscls, mesh.ndims

        # Per-etype element containers and nodal transfer operators
        self.eles, self.gops, self.up2n, self.n2up = {}, {}, {}, {}
        cnodemap, divmap, nsvptsmap = {}, {}, {}
        for et in mesh.spts:
            shapecls = subclass_where(BaseShape, name=et)
            eles = elementscls(shapecls, mesh.spts[et], cfg)
            basis = eles.basis
            nodes = shapecls.std_ele(p)

            # Gradient operator flattened for BLAS; Jacobian in the smats
            jop = basis.ubasis.jac_nodal_basis_at(basis.upts)
            jopf = jop.swapaxes(1, 2).reshape(-1, basis.nupts)
            rcpd = eles.rcpdjac_at_np('upts')
            smatr = eles.smat_at_np('upts')*rcpd[None, :, None, :]

            self.eles[et] = eles
            self.gops[et] = (jopf, smatr)
            nbasis = get_polybasis(et, p, nodes)
            self.up2n[et] = basis.ubasis.nodal_basis_at(nodes)
            self.n2up[et] = nbasis.nodal_basis_at(basis.upts)

            # Corner node numbers key the shared-vdof classes
            snodes = mesh.spts_nodes[et]
            cidxs = shapecls.corner_pts_idxs(snodes.shape[1])
            cnodemap[et] = snodes[:, cidxs]
            divmap[et], nsvptsmap[et] = p, nodes

        shared = np.fromiter(mesh.shared_nodes.by_node, dtype=int)
        self.cleaner = CleanToGrid(cnodemap, divmap, nsvptsmap, shared)

    def grad(self, et, f):
        # Element-local physical gradient at the solution points
        jopf, smatr = self.gops[et]

        refg = interp_pts(jopf, f).reshape(-1, *f.shape)
        return np.einsum('ijkm,ijlm->jlkm', smatr, refg)

    def average(self, fmap):
        # Node-average per-etype (nupts, ..., neles) fields across elements
        shapes, nodal, dtype = {}, {}, first(fmap.values()).dtype
        for et, f in fmap.items():
            fn = interp_pts(self.up2n[et], f)
            shapes[et] = fn.shape
            nodal[et] = fn.reshape(len(fn), -1, fn.shape[-1]).swapaxes(1, 2)

        ncomp = first(nodal.values()).shape[-1]
        avg = self.cleaner.average(nodal, ncomp, dtype)

        out = {}
        for et, f in avg.items():
            remap = self.cleaner.layouts[et][0]
            fe = f[remap].transpose(1, 2, 0).reshape(shapes[et])
            out[et] = interp_pts(self.n2up[et], fe)

        return out

    def _bc_state_info(self):
        # Nothing to recover when there are no boundary faces
        if not self.mesh.bcon:
            return []

        # Boundaries whose classes define a common interface state
        from pyfr.solvers.base import BaseSystem

        scls = subclass_where(BaseSystem,
                              name=self.cfg.get('solver', 'system'))
        bcmap = {b.type: b
                 for b in subclasses(scls.bbcinterscls, just_leaf=True)}

        info = []
        bcsects = self.mesh.bc_sections(self.cfg)
        for bc, con in self.mesh.bcon.items():
            cfgsect = bcsects[bc]
            bcls = bcmap[self.cfg.get(cfgsect, 'type')]
            if hasattr(bcls, 'common_pri_state'):
                c = bcls.common_consts(self.cfg, cfgsect, self.ndims)
                info.append((con, partial(bcls.common_pri_state, c=c)))

        return info

    def _bc_delta(self, out, fmap, con, state, bcrows):
        kind, vidxs = bcrows

        for et, fidx, eidxs in con.items():
            eles, basis = self.eles[et], self.eles[et].basis
            fp = basis.facefpts[fidx]

            # Unit outward normals at the face flux points
            _, _, norm = basis.faces[fidx]
            pn = eles.pnorm_at(basis.fpts[fp], np.tile(norm, (len(fp), 1)))
            pn = pn[:, eidxs].transpose(2, 0, 1)
            nl = pn / np.linalg.norm(pn, axis=0)

            # Interior state at the face flux points
            ul = interp_pts(basis.m0[fp], fmap[et][:, *np.ix_(vidxs, eidxs)])
            ulv = list(ul.swapaxes(0, 1))

            # Evaluate the boundary common state in primitive variables
            if kind == 'con':
                plv = self.elementscls.con_to_pri(ulv, self.cfg)
                urv = self.elementscls.pri_to_con(state(plv, nl), self.cfg)
            else:
                urv = state(ulv, nl)

            out[et][np.ix_(fp, vidxs, eidxs)] = np.stack(urv, axis=1) - ul

    def deltas(self, fmap, favg, bcrows=None):
        # Flux-point jumps between the common values and local states
        out = {}
        for et, f in fmap.items():
            out[et] = interp_pts(self.eles[et].basis.m0, favg[et] - f)

        # Boundaries with known common states refine the deltas
        if bcrows is not None:
            for con, state in self._bc_state_info():
                self._bc_delta(out, fmap, con, state, bcrows)

        return out

    def grad_corrected(self, fmap, bcrows=None):
        # Common-value deltas at the flux points
        dmap = self.deltas(fmap, self.average(fmap), bcrows)

        out = {}
        for et, f in fmap.items():
            basis = self.eles[et].basis
            jopf, smatr = self.gops[et]

            # Reference gradient plus the M6 lift of the deltas
            refg = interp_pts(jopf, f).reshape(-1, *f.shape)
            refg += interp_pts(basis.m6, dmap[et]).reshape(refg.shape)

            # Push forward to the physical gradient with the smats
            out[et] = np.einsum('ijkm,ijlm->jlkm', smatr, refg)

        return out

    def laplacian_of(self, gmap):
        # Stabilised divergence: average, differentiate, average again
        lap = {}
        for et, g in self.average(gmap).items():
            gg = self.grad(et, g.reshape(len(g), -1, g.shape[-1]))
            gg = gg.reshape(*g.shape[:3], *gg.shape[2:])
            lap[et] = np.einsum('ijkkl->ijl', gg)

        return self.average(lap)


def _plugin_fields(soln, fields, pp_plugins):
    # Augment each element type with plugin-derived nodal fields
    derived, dmap = [], {}
    for et, d in soln.data.items():
        fmap = dict(zip(fields, d.swapaxes(0, 1)))
        dv = {n: v for pp in pp_plugins
              for n, v in pp.derived_fields(fmap).items()}
        derived = list(dv)

        # Differentiate the stored and derived fields together
        if dv:
            d = np.concatenate([d, np.stack([*dv.values()], axis=1)], axis=1)

        dmap[et] = d

    return derived, dmap


def _grid_spacing(rec, et):
    # Directional grid spacings from the Jacobian rows
    jinv = rec.gops[et][1]
    hdir = 2/np.linalg.norm(jinv, axis=2)

    return np.stack([hdir.min(axis=0), hdir.max(axis=0)], axis=1)


def _bc_pri_rows(fields, elementscls, ndims, cfg):
    # Locate the mean primitive rows for boundary state corrections
    pnames = [f'avg-{v}' for v in elementscls.privars(ndims, cfg)]
    if all(p in fields for p in pnames):
        return ('pri', [fields.index(p) for p in pnames])
    else:
        return None


def expand_stats_fields(mesh, soln, pp_plugins, elementscls):
    # Augment averaged data with derived fields, gradients and Laplacians
    gridh = any(pp.needs_gridh for pp in pp_plugins)
    grads = any(pp.needs_grads for pp in pp_plugins)

    if not grads and not gridh:
        return

    fields = list(soln.fields)

    # Nodal Laplacians degenerate below second order
    if soln.config.getint('solver', 'order') < 2:
        raise ValueError('Statistics field expansion requires a solution '
                         'of order >= 2')

    rec = FieldRecovery(mesh, elementscls, soln.config)
    bcrows = _bc_pri_rows(fields, elementscls, mesh.ndims, soln.config)
    derived, dmap = _plugin_fields(soln, fields, pp_plugins)

    # Corrected gradients and the associated stabilised Laplacians
    gmap = rec.grad_corrected(dmap, bcrows)
    lmap = rec.laplacian_of(gmap)

    for et, d in dmap.items():
        physg = gmap[et].reshape(len(d), -1, d.shape[-1])
        stack = [d, physg, lmap[et]]
        if gridh:
            stack.append(_grid_spacing(rec, et))

        soln.data[et] = np.concatenate(stack, axis=1)

    # Row names for the augmented layout; names are the contract
    dims = 'xyz'[:mesh.ndims]
    canon = [f.removeprefix('avg-') for f in fields] + derived
    layout = [*fields, *derived,
               *(f'grad_{n}_{x}' for n in canon for x in dims),
               *(f'lap_{n}' for n in canon)]
    if gridh:
        layout += ['grid_hmin', 'grid_hmax']

    soln.derived = derived
    soln.layout = layout
