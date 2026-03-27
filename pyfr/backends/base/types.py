from bisect import insort
from collections import defaultdict, deque
import time

import numpy as np

from pyfr.mpiutil import autofree, mpi


class MatrixBase:
    _base_tags = set()

    def __init__(self, backend, dtype, ioshape, initval, extent, tags):
        self.backend = backend
        self.tags = self._base_tags | tags

        self.dtype = dtype
        self.itemsize = np.dtype(dtype).itemsize

        # Our shape and dimensionality
        shape, ndim = list(ioshape), len(ioshape)

        # SoA and block column size
        soasz, csubsz = backend.soasz, backend.csubsz

        if ndim == 2:
            nrow, ncol = shape

            # Alignment requirement for the leading dimension
            ldmod = csubsz if 'align' in self.tags else 1
            blocked = backend.blocks and 'xchg' not in self.tags
            leaddim = csubsz if blocked else ncol - (ncol % -ldmod)

            nblocks = (ncol - (ncol % -leaddim)) // leaddim
            datashape = [nblocks, nrow, leaddim]
        else:
            nvar, narr, k = shape[-2], shape[-1], soasz
            nparr = narr - narr % -csubsz

            nrow = shape[0] if ndim == 3 else shape[0]*shape[1]
            ncol = nvar*nparr
            leaddim = nvar*csubsz if backend.blocks else ncol

            nblocks = (ncol - (ncol % -leaddim)) // leaddim
            datashape = [nblocks, *shape[:-2], nparr // (nblocks*k), nvar, k]

        # Assign
        self.nrow, self.ncol, self.leaddim = nrow, ncol, leaddim

        self.datashape = datashape
        self.ioshape = ioshape

        self.splitsz = leaddim if backend.blocks else soasz
        self.blocksz = nrow*leaddim
        self.nblocks = nblocks

        self.nbytes = self.nblocks*self.blocksz*self.itemsize
        self.traits = (self.nblocks, nrow, ncol, leaddim, dtype)

        # Process the initial value
        if initval is not None:
            if initval.shape != self.ioshape:
                raise ValueError('Invalid initial value')

            self._initval = np.asanyarray(initval, dtype=self.dtype)
        else:
            self._initval = None

        # Allocate ourself
        backend.malloc(self, extent)

    def get(self):
        # If we are yet to be allocated use our initial value
        if hasattr(self, '_initval'):
            if self._initval is not None:
                return self._initval
            else:
                return np.zeros(self.ioshape, dtype=self.dtype)
        # Otherwise defer to the backend
        else:
            return self._get()

    def _get(self):
        pass

    def _pack(self, ary, out=None):
        # Convert from SoA to [blocked] AoSoA packing
        n, k, csubsz = ary.shape[-1], self.backend.soasz, self.backend.csubsz

        if ary.ndim == 2:
            ary = np.pad(ary, [(0, 0)] + [(0, -n % self.leaddim)])
        else:
            ary = np.pad(ary, [(0, 0)]*(ary.ndim - 1) + [(0, -n % csubsz)])
            ary = ary.reshape(ary.shape[:-1] + (-1, k)).swapaxes(-2, -3)

        ary = ary.reshape(self.nrow, -1, self.leaddim).swapaxes(0, 1)

        if out is not None:
            out.reshape(ary.shape)[:] = ary
            return out

        return np.ascontiguousarray(ary, dtype=self.dtype)

    def _unpack(self, ary):
        # Unpack from blocked AoSoA to blocked SoA
        ary = ary.reshape(self.datashape).swapaxes(-2, -3)

        if len(self.ioshape) > 2:
            ary = np.moveaxis(ary, 0, -3)

        ary = ary.reshape(self.ioshape[:-1] + (-1,))
        ary = ary[..., :self.ioshape[-1]]

        return ary

    def slice(self, ra=None, rb=None, ca=None, cb=None):
        ra, rb = ra or 0, rb or self.nrow
        ca, cb = ca or 0, cb or self.ncol

        return self.backend.matrix_slice(self, ra, rb, ca, cb)


class Matrix(MatrixBase):
    def set(self, ary):
        if ary.shape != self.ioshape:
            raise ValueError('Invalid matrix shape')

        # If we are yet to be allocated then update our initial value
        if hasattr(self, '_initval'):
            self._initval = np.asanyarray(ary, dtype=self.dtype)
        # Otherwise defer to the backend
        else:
            self._set(ary)

    def _set(self, ary):
        pass


class MatrixSlice:
    def __init__(self, backend, mat, ra, rb, ca, cb):
        self.backend = backend
        self.parent = mat

        # Parameter validation
        if ra < 0 or rb > mat.nrow or rb < ra:
            raise ValueError('Invalid row slice')
        if ca < 0 or cb > mat.ncol or cb < ca:
            raise ValueError('Invalid column slice')
        if ca % mat.splitsz != 0:
            raise ValueError('Starting column must conform to backend '
                             'alignment requirements')

        self.ra, self.rb = int(ra), int(rb)
        self.ca, self.cb = int(ca), int(cb)
        self.nrow, self.ncol = self.rb - self.ra, self.cb - self.ca
        self.dtype, self.itemsize = mat.dtype, mat.itemsize
        self.leaddim, self.blocksz = mat.leaddim, mat.blocksz
        self.nblocks = (self.ncol - self.ncol % -self.leaddim) // self.leaddim

        if backend.blocks:
            self.ba, self.bb = self.ca // self.leaddim, self.cb // self.leaddim

        self.traits = (self.nblocks, self.nrow, self.ncol, self.leaddim,
                       self.dtype)

        self.tags = mat.tags | {'slice'}

        # Only set nbytes for slices which are safe to memcpy
        if ca == 0 and cb == mat.ncol:
            self.nbytes = self.nrow*self.leaddim*self.nblocks*self.itemsize

    @property
    def basedata(self):
        return self.parent.basedata

    @property
    def offset(self):
        if self.backend.blocks:
            _offset = self.ba*self.blocksz + self.ra*self.leaddim
        else:
            _offset = self.ra*self.leaddim + self.ca

        return self.parent.offset + _offset*self.itemsize


class ConstMatrix(MatrixBase):
    _base_tags = {'const'}

    def __init__(self, backend, dtype, initval, tags):
        super().__init__(backend, dtype, initval.shape, initval, None, tags)


class XchgMatrix(Matrix):
    _base_tags = {'xchg'}

    def recvreq(self, comm, pid, tag):
        return autofree(comm.Recv_init(self.hdata, pid, tag))

    def sendreq(self, comm, pid, tag):
        return autofree(comm.Send_init(self.hdata, pid, tag))


class View:
    def __init__(self, backend, matmap, rmap, cmap, rstridemap, vshape, tags):
        self.n = len(matmap)
        self.nvrow = vshape[-2] if len(vshape) == 2 else 1
        self.nvcol = vshape[-1] if len(vshape) >= 1 else 1

        # Get the different matrices which we map onto
        self._mats = [backend.mats[i] for i in np.unique(matmap)]

        # Extract the base allocation and data type
        self.basedata = self._mats[0].basedata
        self.refdtype = self._mats[0].dtype

        # Valid matrix types
        mattypes = (backend.matrix_cls, backend.matrix_slice_cls)

        # Validate the matrices
        if any(not isinstance(m, mattypes) for m in self._mats):
            raise TypeError('Incompatible matrix type for view')

        if any(m.basedata != self.basedata for m in self._mats):
            raise TypeError('All viewed matrices must belong to the same '
                            'allocation extent')

        if any(m.dtype != self.refdtype for m in self._mats):
            raise TypeError('Mixed data types are not supported')

        # Index type
        ixdtype = backend.ixdtype

        # SoA size
        k, csubsz = backend.soasz, backend.csubsz

        # Base offsets and leading dimensions for each point
        offset = np.empty(self.n, dtype=ixdtype)
        leaddim = np.empty(self.n, dtype=ixdtype)
        blkdisp = np.empty(self.n, dtype=ixdtype)

        for m in self._mats:
            ix = np.where(matmap == m.mid)
            offset[ix], leaddim[ix] = m.offset // m.itemsize, m.leaddim
            blkdisp[ix] = (cmap[ix]*self.nvcol // m.leaddim)*m.blocksz

        # Row/column displacements
        rowdisp = rmap*leaddim
        cmapmod = cmap % csubsz if backend.blocks else cmap
        coldisp = (cmapmod // k)*k*self.nvcol + cmapmod % k

        mapping = (offset + blkdisp + rowdisp + coldisp)[None, :]
        self.mapping = backend.const_matrix(mapping, dtype=ixdtype, tags=tags)

        # Row strides
        if self.nvrow > 1:
            rstrides = (rstridemap*leaddim)[None, :]
            self.rstrides_val = int(rstrides.flat[0])
            self.rstrides = backend.const_matrix(rstrides, dtype=ixdtype,
                                                 tags=tags)
        else:
            self.rstrides_val = 0
            self.rstrides = None


class XchgView:
    def __init__(self, backend, matmap, rmap, cmap, rstridemap, vshape, tags):
        # Create a normal view
        self.view = backend.view(matmap, rmap, cmap, rstridemap, vshape, tags)

        # Dimensions
        self.n = n = self.view.n
        self.nvrow = nvrow = self.view.nvrow
        self.nvcol = nvcol = self.view.nvcol

        # Now create an exchange matrix to pack the view into
        self.xchgmat = backend.xchg_matrix((nvrow, nvcol*n), tags=tags)

    def recvreq(self, comm, pid, tag):
        return self.xchgmat.recvreq(comm, pid, tag)

    def sendreq(self, comm, pid, tag):
        return self.xchgmat.sendreq(comm, pid, tag)


class Graph:
    def __init__(self, backend):
        self.backend = backend
        self.committed = False

        # Pending kernels and dependencies (buffered until commit)
        self._pkerns = []
        self.kdeps = {}
        self.kpdeps = {}

        # Pending MPI requests
        self._pmpi = []

        # Pending groups
        self._pgroups = []

        # Resolved state (populated during commit)
        self.knodes = {}
        self.depk = set()
        self.mpi_reqs = []
        self.mpi_req_deps = []

        # MPI wrappers
        self._startall = mpi.Prequest.Startall

        if backend.cfg.getbool('backend', 'collect-wait-times', False):
            n = backend.cfg.getint('backend', 'collect-wait-times-len', 10000)
            self._wait_times = wait_times = deque(maxlen=n)

            # Wrap the wait all function with a timing variant
            def waitall(reqs):
                if reqs:
                    t = time.perf_counter_ns()
                    mpi.Prequest.Waitall(reqs)
                    wait_times.append((time.perf_counter_ns() - t) / 1e9)

            self._waitall = waitall
        else:
            self._waitall = mpi.Prequest.Waitall

    def _alldeps(self, k):
        yield from self.kdeps.get(k, [])
        yield from self.kpdeps.get(k, [])

    def add(self, kern, deps=[], pdeps=[]):
        if self.committed:
            raise RuntimeError('Can not add nodes to a committed graph')

        if kern in self.kdeps:
            raise RuntimeError('Can only add a kernel to a graph once')

        self._pkerns.append(kern)
        self.kdeps[kern] = list(deps)
        self.kpdeps[kern] = list(pdeps)

    def add_all(self, kerns, deps=[], pdeps=[]):
        for k in kerns:
            self.add(k, deps, pdeps)

    def add_mpi_req(self, req, deps=[]):
        if self.committed:
            raise RuntimeError('Can not add nodes to a committed graph')

        self._pmpi.append((req, list(deps)))

    def add_mpi_reqs(self, reqs, deps=[]):
        for r in reqs:
            self.add_mpi_req(r, deps)

    def group(self, kerns, subs=[]):
        if self.committed:
            raise RuntimeError('Can not group kernels in a committed graph')

        self._pgroups.append((list(kerns), list(subs)))

    def _build_dag(self):
        # Collapse groups: map each kernel to its group head
        groups = [kerns for kerns, _ in self._pgroups]
        grouped = {}
        for g in groups:
            grouped |= dict.fromkeys(g, g[0])

        to_super = lambda k: grouped.get(k, k)
        gmembers = {g[0]: g for g in groups}

        # Super-node set: ungrouped kernels + group heads
        snodes = [k for k in self.kdeps if k not in grouped or grouped[k] == k]

        # Build adjacency list and in-degree counts
        succs, indeg = defaultdict(list), {}
        for sn in snodes:
            seen = {sn}
            for sk in gmembers.get(sn, [sn]):
                for d in self._alldeps(sk):
                    if (dsn := to_super(d)) not in seen:
                        seen.add(dsn)
                        succs[dsn].append(sn)
            indeg[sn] = len(seen) - 1

        return snodes, succs, indeg, gmembers, to_super

    def _mpi_urgent(self, gmembers, to_super):
        # Walk backwards from MPI send dependencies to find urgent nodes
        urgent = {to_super(d) for _, deps in self._pmpi for d in deps}
        stack = list(urgent)
        while stack:
            sn = stack.pop()
            for sk in gmembers.get(sn, [sn]):
                for d in self._alldeps(sk):
                    if (dsn := to_super(d)) not in urgent:
                        urgent.add(dsn)
                        stack.append(dsn)

        return urgent

    def _topo_sort(self):
        snodes, succs, indeg, gmembers, to_super = self._build_dag()
        urgent = self._mpi_urgent(gmembers, to_super)

        # Priority key: urgent nodes first, then insertion order
        orig = {k: i for i, k in enumerate(self._pkerns)}
        for head, members in gmembers.items():
            orig[head] = min(orig[m] for m in members)
        key = lambda k: (k not in urgent, orig[k])

        # Kahn's algorithm with priority ordering; processing one node
        # at a time gives better interleaving than batch-based approaches
        # since newly-ready urgent nodes are inserted immediately
        ready = sorted((sn for sn in snodes if not indeg[sn]), key=key)
        result = []
        while ready:
            sn = ready.pop(0)
            result.extend(gmembers.get(sn, [sn]))
            for s in succs[sn]:
                indeg[s] -= 1
                if not indeg[s]:
                    insort(ready, s, key=key)

        return result

    def _add_mpi_req(self, req, deps):
        self.mpi_reqs.append(req)
        self.mpi_req_deps.append(deps)

    def _group(self, kerns, subs):
        pass

    def _commit(self):
        pass

    def commit(self):
        self.committed = True

        # Topologically sort all pending kernels
        sorted_kerns = self._topo_sort()
        kern_pos = {k: i for i, k in enumerate(sorted_kerns)}

        # Partition MPI requests into root receives and dep-gated sends
        self.mpi_root_reqs = []
        mpi_at = defaultdict(list)
        for req, deps in self._pmpi:
            if deps:
                mpi_at[max(kern_pos[d] for d in deps)].append((req, deps))
            else:
                self.mpi_root_reqs.append(req)
                self._add_mpi_req(req, deps)

        # Replay kernel adds in sorted order, interleaving MPI sends
        for i, kern in enumerate(sorted_kerns):
            ad = list(self._alldeps(kern))
            self.knodes[kern] = kern.add_to_graph(
                self, [self.knodes[d] for d in ad]
            )
            self.depk.update(ad)

            # Insert MPI sends whose deps are now satisfied
            for req, deps in mpi_at.get(i, []):
                self._add_mpi_req(req, deps)

        # Replay groups (after all kernels are added)
        for kerns, subs in self._pgroups:
            self._group(kerns, subs)

        self._commit()

    def run(self, *args):
        pass

    def get_wait_times(self):
        return list(self._wait_times)
