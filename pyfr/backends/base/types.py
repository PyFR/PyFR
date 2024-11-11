from collections import deque
import time

import numpy as np

from pyfr.mpiutil import autofree, get_comm_rank_root, mpi


class MatrixBase:
    _base_tags = set()

    def __init__(self, backend, dtype, ioshape, initval, extent, aliases,
                 tags):
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

        # Alias or allocate ourself
        if aliases:
            if extent is not None:
                raise ValueError('Aliased matrices can not have an extent')

            backend.alias(self, aliases)
        else:
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

    def _pack(self, ary):
        # Convert from SoA to [blocked] AoSoA packing
        n, k, csubsz = ary.shape[-1], self.backend.soasz, self.backend.csubsz

        if ary.ndim == 2:
            ary = np.pad(ary, [(0, 0)] + [(0, -n % self.leaddim)])
        else:
            ary = np.pad(ary, [(0, 0)]*(ary.ndim - 1) + [(0, -n % csubsz)])
            ary = ary.reshape(ary.shape[:-1] + (-1, k)).swapaxes(-2, -3)

        ary = ary.reshape(self.nrow, -1, self.leaddim).swapaxes(0, 1)

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
    def __init__(self, backend, dtype, ioshape, initval, extent, aliases,
                 tags):
        super().__init__(backend, dtype, ioshape, initval, extent, aliases,
                         tags)

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
        super().__init__(backend, dtype, initval.shape, initval,
                         None, None, tags)


class XchgMatrix(Matrix):
    _base_tags = {'xchg'}

    def recvreq(self, pid, tag):
        comm, rank, root = get_comm_rank_root()

        return autofree(comm.Recv_init(self.hdata, pid, tag))

    def sendreq(self, pid, tag):
        comm, rank, root = get_comm_rank_root()

        return autofree(comm.Send_init(self.hdata, pid, tag))


class View:
    def __init__(self, backend, matmap, rmap, cmap, rstridemap, vshape, tags):
        self.n = len(matmap)
        self.nvrow = vshape[-2] if len(vshape) == 2 else 1
        self.nvcol = vshape[-1] if len(vshape) >= 1 else 1
        self.rstrides = None

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
            self.rstrides = backend.const_matrix(rstrides, dtype=ixdtype,
                                                 tags=tags)


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

    def recvreq(self, pid, tag):
        return self.xchgmat.recvreq(pid, tag)

    def sendreq(self, pid, tag):
        return self.xchgmat.sendreq(pid, tag)


class Graph:
    def __init__(self, backend):
        self.backend = backend
        self.committed = False

        # Kernels and their dependencies
        self.knodes = {}
        self.kdeps = {}
        self.depk = set()

        # Grouped kernels
        self.groupk = set()

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

        # MPI requests along with their associated dependencies
        self.mpi_reqs = []
        self.mpi_req_deps = []

    def add(self, kern, deps=[], pdeps=[]):
        if self.committed:
            raise RuntimeError('Can not add nodes to a committed graph')

        if kern in self.knodes:
            raise RuntimeError('Can only add a kernel to a graph once')

        # Handle priority-enforcing (false) dependencies
        adeps = [*deps, *pdeps] if self.needs_pdeps else deps

        # Resolve the dependency list
        rdeps = [self.knodes[d] for d in adeps]

        # Ask the kernel to add itself
        self.knodes[kern] = kern.add_to_graph(self, rdeps)

        # Note our dependencies
        self.kdeps[kern] = list(deps)
        self.depk.update(deps)

    def add_all(self, kerns, deps=[], pdeps=[]):
        for k in kerns:
            self.add(k, deps, pdeps)

    def add_mpi_req(self, req, deps=[]):
        if self.committed:
            raise RuntimeError('Can not add nodes to a committed graph')

        if req in self.mpi_reqs:
            raise ValueError('Can only add an MPI request to a graph once')

        # Add the request
        self.mpi_reqs.append(req)
        self.mpi_req_deps.append(deps)

        # Note any dependencies
        self.depk.update(deps)

    def add_mpi_reqs(self, reqs, deps=[]):
        for r in reqs:
            self.add_mpi_req(r, deps)

    def _iter_deps(self, kern):
        for d in self.kdeps[kern]:
            yield d
            yield from self._iter_deps(d)

    def group(self, kerns, subs=[]):
        if self.committed:
            raise RuntimeError('Can not group kernels in a committed graph')

        klist = list(kerns)
        kset = set(klist)

        # Ensure kernels are only in a single group
        if not self.groupk.isdisjoint(kset):
            raise ValueError('Kernels can only be in one group')

        # Validate the dependencies of the grouping
        for i, k in enumerate(klist):
            for d in self.kdeps[k]:
                if d in klist[i + 1:]:
                    raise ValueError('Inconsistent kernel grouping order')

                if d not in kset and not kset.isdisjoint(self._iter_deps(d)):
                    raise ValueError('Kernel grouping violates dependencies')

        # Ensure the substitutions are consistent with the grouping
        if any(k not in klist for s in subs for k, v in s):
            raise ValueError('Invalid kernels in substitution list')

        # Mark these kernels as being grouped
        self.groupk.update(kerns)

    def commit(self):
        mreqs, mdeps = self.mpi_reqs, self.mpi_req_deps

        self.committed = True
        self.mpi_root_reqs = [r for r, d in zip(mreqs, mdeps) if not d]

    def run(self, *args):
        pass

    def get_wait_times(self):
        return list(self._wait_times)
