# -*- coding: utf-8 -*-

import numpy as np


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
            leaddim = csubsz if backend.blocks else ncol - (ncol % -ldmod)

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
    def __init__(self, backend, ioshape, initval, extent, aliases, tags):
        super().__init__(backend, backend.fpdtype, ioshape, initval, extent,
                         aliases, tags)

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

    def __init__(self, backend, initval, extent, tags):
        super().__init__(backend, backend.fpdtype, initval.shape, initval,
                         extent, None, tags)


class XchgMatrix(Matrix):
    def recvreq(self, pid, tag):
        from mpi4py import MPI

        return MPI.COMM_WORLD.Recv_init(self.hdata, pid, tag)

    def sendreq(self, pid, tag):
        from mpi4py import MPI

        return MPI.COMM_WORLD.Send_init(self.hdata, pid, tag)


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

        # SoA size
        k, csubsz = backend.soasz, backend.csubsz

        # Base offsets and leading dimensions for each point
        offset = np.empty(self.n, dtype=np.int32)
        leaddim = np.empty(self.n, dtype=np.int32)
        blkdisp = np.empty(self.n, dtype=np.int32)

        for m in self._mats:
            ix = np.where(matmap == m.mid)
            offset[ix], leaddim[ix] = m.offset // m.itemsize, m.leaddim
            blkdisp[ix] = (cmap[ix]*self.nvcol // m.leaddim)*m.blocksz

        # Row/column displacements
        rowdisp = rmap*leaddim
        cmapmod = cmap % csubsz if backend.blocks else cmap
        coldisp = (cmapmod // k)*k*self.nvcol + cmapmod % k

        mapping = (offset + blkdisp + rowdisp + coldisp)[None, :]
        self.mapping = backend.base_matrix_cls(
            backend, np.int32, (1, self.n), mapping, None, None, tags
        )

        # Row strides
        if self.nvrow > 1:
            rstrides = (rstridemap*leaddim)[None, :]
            self.rstrides = backend.base_matrix_cls(
                backend, np.int32, (1, self.n), rstrides, None, None, tags
            )


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


class Queue:
    def __init__(self, backend):
        from mpi4py import MPI

        self.backend = backend

        # MPI wrappers
        self._startall = MPI.Prequest.Startall
        self._waitall = MPI.Prequest.Waitall

        # Items waiting to be executed
        self._items = []

    def enqueue(self, items, *args, **kwargs):
        self._items.extend((item, args, kwargs) for item in items)

    def enqueue_and_run(self, items, *args, **kwargs):
        if self._items:
            self.run()

        self.enqueue(items, *args, **kwargs)
        self.run()

    def __bool__(self):
        return bool(self._items)
