# -*- coding: utf-8 -*-

import pyfr.backends.base as base

import numpy as np

class _Dummy2DBase(object):
    def __init__(self, nrow, ncol, initval=None, tags=set()):
        self._nrow = nrow
        self._ncol = ncol
        self._tags = tags
        self._data = initval if initval is not None else np.empty((nrow, ncol))

    def get(self):
        return self._data

    @property
    def nrow(self):
        return self._nrow

    @property
    def ncol(self):
        return self._ncol

    @property
    def nbytes(self):
        return self._data.nbytes


class _DummyMatrix(_Dummy2DBase, base.Matrix):
    def set(self, arr):
        assert arr.shape == (self._nrow, self._ncol)
        self._data = arr


class _DummyConstMatrix(_Dummy2DBase, base.ConstMatrix):
    def __init__(self, initval, tags=set()):
        nrow, ncol = initval.shape
        super(_DummyConstMatrix, self).__init__(nrow, ncol, initval, tags)


class _DummySparseMatrix(_Dummy2DBase, base.SparseMatrix):
    def __init__(self, initval, tags=set()):
        nrow, ncol = initval.shape
        super(_DummySparseMatrix, self).__init__(nrow, ncol, initval, tags)


class _DummyMPIMatrix(_DummyMatrix, base.MPIMatrix):
    pass


class _DummyMatrixBank(base.MatrixBank):
    def __init__(self, nrow, ncol, nbanks, initval=None, tags=set()):
        mats = [_DummyMatrix(nrow, ncol, initval, tags) for i in xrange(nbanks)]
        super(_DummyMatrixBank, self).__init__(mats)


class _DummyView(base.View):
    def __init__(self, matmap, rcmap, tags):
        pass


class _DummyMPIView(base.MPIView):
    def __init__(self, matmap, rcmap, tags):
        pass


class _DummyKernel(base.Kernel):
    pass


class _DummyQueue(base.Queue):
    def __lshift__(self, iterable):
        pass

    def __mod__(self, iterable):
        pass

class DummyBackend(base.Backend):
    def _matrix(self, *args, **kwargs):
        return _DummyMatrix(*args, **kwargs)

    def _matrix_bank(self, *args, **kwargs):
        return _DummyMatrixBank(*args, **kwargs)

    def _mpi_matrix(self, *args, **kwargs):
        return _DummyMPIMatrix(*args, **kwargs)

    def _const_matrix(self, *args, **kwargs):
        return _DummyConstMatrix(*args, **kwargs)

    def _sparse_matrix(self, *args, **kwargs):
        return _DummySparseMatrix(*args, **kwargs)

    def _is_sparse(self, mat, tags):
        if 'sparse' in tags:
            return True
        elif np.count_nonzero(mat) < 0.33*mat.size:
            return True
        else:
            return False

    def _view(self, *args, **kwargs):
        return _DummyView(*args, **kwargs)

    def _mpi_view(self, *args, **kwargs):
        return _DummyMPIView(*args, **kwargs)

    def _kernel(self, name, *args, **kwargs):
        validateattr = '_validate_' + name

        # Check the kernel name
        if not hasattr(self, validateattr):
            raise PyFRInvalidKernelError("'{}' is not a valid kernel"\
                                         .format(name))

        # Call the validator method to check the arguments
        getattr(self, validateattr)(*args, **kwargs)

        return _DummyKernel()

    def _queue(self, *args, **kwargs):
        return _DummyQueue(*args, **kwargs)

    def runall(self, seq):
        for q in seq:
            assert isinstance(q, _DummyQueue)

    def _validate_mul(self, a, b, out, alpha=None, beta=None):
        assert a.nrow == out.nrow
        assert a.ncol == b.nrow
        assert b.ncol == out.ncol
        assert isinstance(out, (_DummyMatrix, _DummyMatrixBank))

    def _validate_ipadd(self, y, alpha, x):
        assert isinstance(out, (_DummyMatrix, _DummyMatrixBank))
