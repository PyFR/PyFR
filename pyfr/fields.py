from collections import defaultdict, namedtuple
from functools import partial
import itertools as it

import numpy as np

from pyfr.bases import get_basis
from pyfr.cache import memoize
from pyfr.metrics import MeshMetrics
from pyfr.mpiutil import (DistributedDirectory, first_coll,
                          get_comm_rank_root, mpi)
from pyfr.nputil import zip_chunks
from pyfr.quadrules import get_quadrule
from pyfr.readers.native import group_names
from pyfr.shapes import BaseShape, apply_op, proj_pts
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
    # Accumulation chunk size, in items
    CHUNK_SZ = 2**15

    # Batch width for averaging multiple items, in components
    BATCH_NCOMP = 32

    @staticmethod
    def _class_rows(classes, nint):
        return (classes[:, None]*nint + np.arange(nint)).ravel()

    def __init__(self, cnodemap, divmap, nsvptsmap, shared):
        self.comm, _, _ = get_comm_rank_root()
        self.cnodemap = cnodemap

        topos, ktypes = {}, set()
        for etype in cnodemap:
            sdiv = get_subdiv(etype, divmap[etype])
            topos[etype] = sdiv.topology(nsvptsmap[etype])
            ktypes.update(topos[etype].kinds)

        # Reduce to obtain the global set of entities
        ktypes = set().union(*self.comm.allgather(ktypes))

        # Per-kind cross-etype dedup -> self.kinds (with emap populated)
        self.kinds = {kind: self._init_kind(kind, topos, cnodemap, shared)
                      for kind in sorted(ktypes)}

        self.layouts = {etype: self._build_layout(etype, topos[etype])
                        for etype in cnodemap}

    def _init_kind(self, kind, topos, cnodemap, shared):
        evdofs, klist, kinfo = {}, [], None
        for etype, cn in cnodemap.items():
            if kind not in topos[etype].kinds:
                continue

            keys, fsrc, cpos, nint = topos[etype].canonical_vdofs(kind, cn)
            evdofs[etype] = (keys, fsrc, cpos)
            klist.append(keys)
            kinfo = (nint, keys.shape[1])

        # Agree on nint and key width across ranks
        nint, ncols = first_coll(self.comm, kinfo)

        # Deduplicate canonical keys across all contributing etypes
        akeys = np.concatenate(klist or [np.empty((0, ncols), dtype=int)])
        uniq, inv = _unique_rows(akeys)

        # Select the classes which must be reduced across ranks
        defidx = np.flatnonzero(np.isin(uniq, shared).all(axis=1))

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
        remap = np.empty((npts, neles), dtype=int)
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
            remap.ravel()[fsrc] = off + within
            kept.append(ksrc)
            off += len(classes)*g.nint

        # Body vdofs: unique by construction (sequential kept ids)
        bidxs = topo.body_idxs
        bpos = (bidxs[:, None]*neles + np.arange(neles)).ravel()
        remap[bidxs] = off + np.arange(bpos.size).reshape(-1, neles)
        kept.append(bpos)

        return remap.T, np.concatenate(kept), bpos

    @memoize
    def _scatter_plan(self, kname):
        # Rows sharing an entry count are gathered at a common width
        g = self.kinds[kname]

        nrows, plan = len(g.keys)*g.nint, {}
        for etype, (_, fsrc, fdst) in g.emap.items():
            # Sorting by destination brings each row's entries together
            counts = np.bincount(fdst, minlength=nrows)
            sfsrc = fsrc[np.argsort(fdst, kind='stable')]
            starts = np.cumsum(counts) - counts

            plan[etype] = blocks = []
            for nfan in np.unique(counts):
                if nfan == 0:
                    continue

                # Here cols[i, j] is the jth entry of row rows[i]
                rows = np.flatnonzero(counts == nfan)
                cols = sfsrc[starts[rows][:, None] + np.arange(nfan)]
                blocks.append((rows, cols))

        return plan

    def _scatter(self, psum, src, blocks, csize):
        buf = np.empty((csize, src.shape[1]), dtype=src.dtype)

        for rows, cols in blocks:
            for rs, cs in zip_chunks(csize, rows, cols):
                # The first slot seeds the sum and the rest add into it
                head, *tail = cs.T
                acc = np.take(src, head, axis=0, mode='clip')
                tmp = buf[:len(rs)]
                for col in tail:
                    np.take(src, col, axis=0, out=tmp, mode='clip')
                    acc += tmp

                psum[rs] += acc

    def select(self, etype, values):
        _, kept, _ = self.layouts[etype]
        flat = values.reshape(-1, *values.shape[2:])
        return flat[kept]

    def average(self, fields, ncomp, dtype):
        # Per-kind: accumulate sums -> exchange deficient sums -> divide by
        # precomputed topological count
        vdofvals = {etype: vals.reshape(-1, ncomp)
                    for etype, vals in fields.items()}

        # Chunking keeps the accumulator resident across the gather slots
        csize = max(1, self.CHUNK_SZ // ncomp)

        out = {etype: [] for etype in fields}
        for kname, g in self.kinds.items():
            psum = np.zeros((len(g.keys)*g.nint, ncomp), dtype=dtype)
            plan = self._scatter_plan(kname)

            # Non-finite values from guarded ratios propagate by design
            with np.errstate(invalid='ignore'):
                for etype in g.emap:
                    self._scatter(psum, vdofvals[etype], plan[etype], csize)

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

    def average_items(self, items):
        # Agree on the component count of each item across ranks
        widths = [max((a.shape[2] for a in ef.values()), default=1)
                  for ef, _ in items]
        ncomps = np.array(widths, dtype=int)
        self.comm.Allreduce(mpi.IN_PLACE, ncomps, op=mpi.MAX)

        # Group items by data type into bounded-width batches
        batches = defaultdict(list)
        for i, ((_, dt), w) in enumerate(zip(items, ncomps)):
            grp = batches[np.dtype(dt).type]
            if not grp or ncomps[grp[-1]].sum() + w > self.BATCH_NCOMP:
                grp.append([])

            grp[-1].append(i)

        # Average each batch and split into per-item outputs
        out = [None]*len(items)
        for dt, grp in batches.items():
            for ids in grp:
                offs = list(it.accumulate(ncomps[ids], initial=0))
                efs = [items[i][0] for i in ids]

                bef = {}
                for et in efs[0]:
                    parts = [ef[et] for ef in efs]
                    bef[et] = np.concatenate(parts, axis=2, dtype=dt)

                avg = self.average(bef, offs[-1], dt)
                for i, (a, b) in zip(ids, it.pairwise(offs)):
                    out[i] = {et: v[:, a:b] for et, v in avg.items()}

        return out


def con_block_to_pri(elementscls, cfg, ndims, block, groups=()):
    # Fields-first conservative block (+grads, +resid) to primitives
    nvars = len(elementscls.convars(ndims, cfg))
    fields = elementscls.con_to_pri(block[:nvars], cfg)

    # Solution gradients convert via the chain rule
    if 'grad' in groups:
        ng = nvars*(1 + ndims)
        dcon = block[nvars:ng].reshape(nvars, ndims, *block.shape[1:])
        dpri = elementscls.diff_con_to_pri(block[:nvars], dcon, cfg)

        fields += [f for gf in dpri for f in gf]

    # Residuals convert to primitive rates
    if 'resid' in groups:
        fields += elementscls.diff_con_to_pri(block[:nvars], block[-nvars:],
                                              cfg)

    return np.array(fields)


class FieldRecovery:
    # Differentiated block chunk size, in items
    CHUNK_SZ = 2**21

    def __init__(self, mesh, elementscls, cfg):
        self.order = p = cfg.getint('solver', 'order')
        self.mesh, self.cfg = mesh, cfg
        self.elementscls, self.ndims = elementscls, mesh.ndims

        # Per-etype metric terms and nodal transfer operators
        self.metrics, self.gops, self.fops = {}, {}, {}
        self.up2n, self.n2up, self.n2g = {}, {}, {}
        cnodemap, divmap, nsvptsmap = {}, {}, {}
        for et in mesh.spts:
            shapecls = subclass_where(BaseShape, name=et)
            shape = shapecls(len(mesh.spts[et]), cfg)
            metrics = MeshMetrics(shape, mesh.spts[et], mesh.spts_curved[et])
            nodes = shapecls.std_ele(p)

            # Gradient operator flattened for BLAS; J^-1 from the smats
            jac = shape.ubasis.jac_nodal_basis_at(shape.upts)
            jacop = jac.swapaxes(1, 2).reshape(-1, shape.nupts)
            rcpd = metrics.rcpdjac_at('upts')
            jinv = metrics.smat_at('upts')*rcpd[None, :, None, :]

            self.metrics[et] = metrics
            self.gops[et] = (jacop, jinv)

            # Build nodal transfer operators for the boundary nodes
            nbasis = get_basis(et, p, nodes)
            bmask = np.ones(len(nodes), dtype=bool)
            bmask[get_subdiv(et, p).topology(nodes).body_idxs] = False
            bnodes = nodes[bmask]
            self.up2n[et] = shape.ubasis.nodal_basis_at(bnodes)
            self.n2up[et] = nbasis.nodal_basis_at(shape.upts)[:, bmask]
            self.n2g[et], self.fops[et] = self._face_ops(shape, bnodes)

            # Corner node numbers key the shared-vdof classes
            snodes = mesh.spts_nodes[et]
            cidxs = shapecls.corner_pts_idxs(snodes.shape[1])
            cnodemap[et] = snodes[:, cidxs]
            divmap[et], nsvptsmap[et] = p, bnodes

        shared = np.fromiter(mesh.shared_nodes.by_node, dtype=int)
        self.cleaner = CleanToGrid(cnodemap, divmap, nsvptsmap, shared)

    @staticmethod
    def _face_ops(shape, bnodes):
        p, ubasis = shape.order, shape.ubasis
        ou = ubasis.ortho_basis_at(shape.upts)

        # Build surface lifting and quadrature operators for each face
        n2g, fops = np.zeros((shape.ndims*shape.nupts, len(bnodes))), []
        for kind, proj, norm in shape.faces:
            fnodes = subclass_where(BaseShape, name=kind).std_ele(p)
            pfnodes = proj_pts(proj, fnodes)
            fb = np.linalg.norm(bnodes - pfnodes[:, None], axis=2).argmin(1)

            qr = get_quadrule(kind, qdeg=max(2*p, 1))
            qpts = proj_pts(proj, qr.pts)
            oq = ou.T @ (ubasis.ortho_basis_at(qpts)*qr.wts)
            q2g = (np.asarray(norm)[:, None, None]*oq).reshape(-1, len(qpts))
            fn2q = get_basis(kind, p, fnodes).nodal_basis_at(qr.pts)

            n2g[:, fb] += q2g @ fn2q
            fops.append((qpts, fb, ubasis.nodal_basis_at(qpts), fn2q, q2g))

        return n2g, fops

    def _nodal_deltas(self, fmap):
        # Interpolate fields to the element boundary nodes
        nodal = {}
        for et, f in fmap.items():
            fn = apply_op(self.up2n[et], f)
            nodal[et] = fn.reshape(len(fn), -1, fn.shape[-1]).swapaxes(1, 2)

        # Component count and data type, from any rank holding elements
        ninfo = first(((n.shape[-1], n.dtype) for n in nodal.values()), None)
        ncomp, dtype = first_coll(self.cleaner.comm, ninfo)
        avg = self.cleaner.average(nodal, ncomp, dtype)

        # Compute the common-value jumps at the boundary nodes
        out = {}
        for et, f in fmap.items():
            nd = avg[et][self.cleaner.layouts[et][0].T] - nodal[et]
            out[et] = nd.swapaxes(1, 2).reshape(-1, *f.shape[1:])

        return out

    def average(self, fmap):
        # Node-average per-etype (nupts, ..., neles) fields across elements
        ndeltas = self._nodal_deltas(fmap)

        # Lift the boundary-node jumps to the solution points
        return {et: f + apply_op(self.n2up[et], ndeltas[et])
                for et, f in fmap.items()}

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

    def _bc_grad(self, refg, fmap, ndeltas, con, state, bcrows):
        kind, vidxs = bcrows

        for et, fidx, eidxs in con.items():
            qpts, fb, u2q, fn2q, q2g = self.fops[et][fidx]

            # Unit outward normals at the face quadrature points
            pn = self.metrics[et].pnorm_at(qpts, fidx)
            pn = pn[:, eidxs].transpose(2, 0, 1)
            nl = pn / np.linalg.norm(pn, axis=0)

            # Interior state at the face quadrature points
            ul = apply_op(u2q, fmap[et][:, *np.ix_(vidxs, eidxs)])
            ulv = list(ul.swapaxes(0, 1))

            # Evaluate the boundary common state in primitive variables
            if kind == 'con':
                plv = self.elementscls.con_to_pri(ulv, self.cfg)
                urv = self.elementscls.pri_to_con(state(plv, nl), self.cfg)
            else:
                urv = state(ulv, nl)

            # Lift the residual boundary jump into the transformed gradients
            nd = apply_op(fn2q, ndeltas[et][np.ix_(fb, vidxs, eidxs)])
            ur = np.stack(urv, axis=1).astype(ul.dtype, copy=False)
            dg = apply_op(q2g, ur - ul - nd)
            dg = dg.reshape(self.ndims, -1, *ur.shape[1:])
            refg[et][:, :, *np.ix_(vidxs, eidxs)] += dg

    def grad_corrected(self, fmap, bcrows=None):
        # Common-value deltas at the boundary nodes
        ndeltas = self._nodal_deltas(fmap)

        # Compute the corrected gradients in transformed space
        refg = {}
        for et, f in fmap.items():
            rg = apply_op(self.gops[et][0], f)
            rg += apply_op(self.n2g[et], ndeltas[et])
            refg[et] = rg.reshape(self.ndims, *f.shape)

        # Refine the gradients on boundaries with known common states
        if bcrows is not None:
            for con, state in self._bc_state_info():
                self._bc_grad(refg, fmap, ndeltas, con, state, bcrows)

        # Transform the gradients from reference to physical space
        return {et: np.einsum('ijkm,ijlm->jlkm', self.gops[et][1], rg)
                for et, rg in refg.items()}

    def laplacian_of(self, gmap):
        # Stabilised divergence: average, differentiate, average again
        lap = {}
        for et, g in self.average(gmap).items():
            jacop, jinv = self.gops[et]
            nupts, neles = len(g), g.shape[-1]
            gf = g.reshape(nupts, -1, neles)

            # Chunked so the differentiated block never reaches memory
            csize = max(1, self.CHUNK_SZ // (len(jacop)*gf.shape[1]))

            lap[et] = np.empty((nupts, g.shape[1], neles), g.dtype)
            chunks = zip_chunks(csize, lap[et], gf, jinv, axis=-1)

            for lchunk, gchunk, jchunk in chunks:
                refg = apply_op(jacop, gchunk)
                refg = refg.reshape(-1, *g.shape[:-1], lchunk.shape[-1])
                np.einsum('ijkm,ijlkm->jlm', jchunk, refg, out=lchunk)

        return self.average(lap)


def recover_grads(mesh, soln, elementscls):
    rec = FieldRecovery(mesh, elementscls, soln.config)
    nvars = len(soln.fields)
    bcvars = ('con', list(range(nvars)))
    data = soln.data

    # Insert the gradients between the fields and any residuals
    fmap = {et: d[:, :nvars] for et, d in data.items()}
    for et, g in rec.grad_corrected(fmap, bcvars).items():
        d = data[et]
        g = g.reshape(len(d), -1, d.shape[-1])
        data[et] = np.concatenate([d[:, :nvars], g, d[:, nvars:]], axis=1)

    gnames = group_names(soln.fields, mesh.ndims, {'grad'})
    soln.rownames[nvars:nvars] = gnames
    soln.groups.add('grad')


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
        return []

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

    # Row names for the augmented data
    dims = 'xyz'[:mesh.ndims]
    canon = [f.removeprefix('avg-') for f in fields] + derived
    rownames = [*fields, *derived,
                *(f'grad_{n}_{x}' for n in canon for x in dims),
                *(f'lap_{n}' for n in canon)]
    if gridh:
        rownames += ['grid_hmin', 'grid_hmax']

    soln.derived = derived
    soln.rownames = rownames

    # Report the names of the rows appended to the stored fields
    return rownames[len(fields):]
