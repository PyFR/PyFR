# -*- coding: utf-8 -*-

from collections import Sequence, deque

import numpy as np


class MatrixBase(object):
    _base_tags = set()

    def __init__(self, backend, dtype, ioshape, initval, extent, aliases,
                 tags):
        self.backend = backend
        self.tags = self._base_tags | tags

        self.dtype = dtype
        self.itemsize = np.dtype(dtype).itemsize

        # Alignment requirement for the leading dimension
        ldmod = backend.alignb // self.itemsize if 'align' in tags else 1

        # Our shape and dimensionality
        shape, ndim = list(ioshape), len(ioshape)

        if ndim == 2:
            nrow, ncol = shape
            aosoashape = shape
        else:
            nvar, narr, k = shape[-2], shape[-1], backend.soasz
            nparr = narr - narr % -k

            nrow = shape[0] if ndim == 3 else shape[0]*shape[1]
            ncol = nvar*nparr
            aosoashape = shape[:-2] + [nparr // k, nvar, k]

        # Assign
        self.nrow, self.ncol = int(nrow), int(ncol)

        self.datashape = aosoashape
        self.ioshape = ioshape

        self.leaddim = self.ncol - (self.ncol % -ldmod)

        self.pitch = self.leaddim*self.itemsize
        self.nbytes = self.nrow*self.pitch
        self.traits = (self.nrow, self.ncol, self.leaddim, self.dtype)

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
        # If necessary convert from SoA to AoSoA packing
        if ary.ndim > 2:
            n, k = ary.shape[-1], self.backend.soasz

            ary = np.pad(ary, [(0, 0)]*(ary.ndim - 1) + [(0, -n % k)],
                         mode='constant')
            ary = ary.reshape(ary.shape[:-1] + (-1, k)).swapaxes(-2, -3)
            ary = ary.reshape(self.nrow, self.ncol)

        return np.ascontiguousarray(ary, dtype=self.dtype)

    def _unpack(self, ary):
        # If necessary unpack from AoSoA to SoA
        if len(self.ioshape) > 2:
            ary = ary.reshape(self.datashape)
            ary = ary.swapaxes(-2, -3)
            ary = ary.reshape(self.ioshape[:-1] + (-1,))
            ary = ary[..., :self.ioshape[-1]]

        return ary

    def slice(self, ra=None, rb=None, ca=None, cb=None):
        ra, rb = ra or 0, rb or self.nrow
        ca, cb = ca or 0, cb or self.ncol

        return self.backend.matrix_slice(self, ra, rb, ca, cb)


class Matrix(MatrixBase):
    _base_tags = {'dense'}

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


class MatrixSlice(object):
    def __init__(self, backend, mat, ra, rb, ca, cb):
        self.backend = backend
        self.parent = mat
        self.datamap = {}

        # Parameter validation
        if ra < 0 or rb > mat.nrow or rb < ra:
            raise ValueError('Invalid row slice')
        if ca < 0 or cb > mat.ncol or cb < ca:
            raise ValueError('Invalid column slice')
        if ca*mat.itemsize % backend.alignb != 0:
            raise ValueError('Starting column must conform to backend '
                             'alignment requirements')
        if isinstance(mat, MatrixBank) and any('bank' in m.tags for m in mat):
            raise TypeError('Nested MatrixBank objects can not be sliced')

        self.ra, self.rb = int(ra), int(rb)
        self.ca, self.cb = int(ca), int(cb)
        self.nrow, self.ncol = self.rb - self.ra, self.cb - self.ca
        self.dtype, self.itemsize = mat.dtype, mat.itemsize
        self.leaddim, self.pitch = mat.leaddim, mat.pitch

        self.traits = (self.nrow, self.ncol, self.leaddim, self.dtype)
        self.tags = mat.tags | {'slice'}

        # Only set nbytes for slices which are safe to memcpy
        if ca == 0 and cb == mat.ncol:
            self.nbytes = self.nrow*self.pitch

    @property
    def basedata(self):
        if 'bank' in self.tags:
            raise AttributeError('basedata undefined for banked slices')

        return self.parent.basedata

    @property
    def offset(self):
        if 'bank' in self.tags:
            raise AttributeError('offset undefined for banked slices')

        return self.parent.offset + self.ra*self.pitch + self.ca*self.itemsize

    @property
    def data(self):
        try:
            return self.datamap[self.parent.mid]
        except KeyError:
            data = self._init_data(self.parent)
            self.datamap[self.parent.mid] = data

            return data


class ConstMatrix(MatrixBase):
    _base_tags = {'const', 'dense'}

    def __init__(self, backend, initval, extent, tags):
        super().__init__(backend, backend.fpdtype, initval.shape, initval,
                         extent, None, tags)


class XchgMatrix(Matrix):
    pass


class MatrixBank(Sequence):
    def __init__(self, backend, mats, initbank, tags):
        mats = list(mats)

        # Ensure all matrices have the same traits
        if any(m.traits != mats[0].traits for m in mats[1:]):
            raise ValueError('Matrices in a bank must be homogeneous')

        # Check that all matrices share tags
        if any(m.tags != mats[0].tags for m in mats[1:]):
            raise ValueError('Matrices in a bank must share tags')

        self.backend = backend
        self.tags = tags | mats[0].tags | {'bank'}

        self._mats = mats
        self._curr_idx = initbank
        self._curr_mat = mats[initbank]

    def __len__(self):
        return len(self._mats)

    def __getitem__(self, idx):
        return self._mats[idx]

    def __getattr__(self, attr):
        return getattr(self._curr_mat, attr)

    slice = MatrixBase.slice

    @property
    def active(self):
        return self._curr_idx

    @active.setter
    def active(self, idx):
        self._curr_idx = idx
        self._curr_mat = self._mats[idx]


class View(object):
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
        if any(not isinstance(m, mattypes) or 'bank' in m.tags
               for m in self._mats):
            raise TypeError('Incompatible matrix type for view')

        if any(m.basedata != self.basedata for m in self._mats):
            raise TypeError('All viewed matrices must belong to the same '
                            'allocation extent')

        if any(m.dtype != self.refdtype for m in self._mats):
            raise TypeError('Mixed data types are not supported')

        # SoA size
        k = backend.soasz

        # Base offsets and leading dimensions for each point
        offset = np.empty(self.n, dtype=np.int32)
        leaddim = np.empty(self.n, dtype=np.int32)

        for m in self._mats:
            ix = np.where(matmap == m.mid)
            offset[ix], leaddim[ix] = m.offset // m.itemsize, m.leaddim

        # Row/column displacements
        rowdisp = rmap*leaddim
        coldisp = (cmap // k)*(self.nvcol*k) + cmap % k

        mapping = (offset + rowdisp + coldisp)[None,:]
        self.mapping = backend.base_matrix_cls(
            backend, np.int32, (1, self.n), mapping, None, None, tags
        )

        # Row strides
        if self.nvrow > 1:
            rstrides = (rstridemap*leaddim)[None,:]
            self.rstrides = backend.base_matrix_cls(
                backend, np.int32, (1, self.n), rstrides, None, None, tags
            )


class XchgView(object):
    def __init__(self, backend, matmap, rmap, cmap, rstridemap, vshape, tags):
        # Create a normal view
        self.view = backend.view(matmap, rmap, cmap, rstridemap, vshape, tags)

        # Dimensions
        self.n = n = self.view.n
        self.nvrow = nvrow = self.view.nvrow
        self.nvcol = nvcol = self.view.nvcol

        # Now create an exchange matrix to pack the view into
        self.xchgmat = backend.xchg_matrix((nvrow, nvcol*n), tags=tags)


class Queue(object):
    def __init__(self, backend):
        self.backend = backend

        # Last kernel we executed
        self._last = None

        # Items waiting to be executed
        self._items = deque()

        # Active MPI requests
        self.mpi_reqs = []

    def __lshift__(self, items):
        self._items.extend(items)

    def __mod__(self, items):
        self.run()
        self << items
        self.run()

    def __bool__(self):
        return bool(self._items)

    def run(self):
        while self._items:
            self._exec_next()
        self._wait()

    def _exec_item(self, item, args, kwargs):
        item.run(self, *args, **kwargs)
        self._last = item

    def _exec_next(self):
        item, args, kwargs = self._items.popleft()

        # If we are at a sequence point then wait for current items
        if self._at_sequence_point(item):
            self._wait()

        # Execute the item
        self._exec_item(item, args, kwargs)

    def _exec_nowait(self):
        while self._items and not self._at_sequence_point(self._items[0][0]):
            self._exec_item(*self._items.popleft())

    def _at_sequence_point(self, item):
        pass

    def _wait(self):
        pass
