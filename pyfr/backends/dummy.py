# -*- coding: utf-8 -*-

import pyfr.backends.base as base

import numpy as np

class DummyMatrixBase(base.MatrixBase):
    def __init__(self, backend, ioshape, initval, iopacking, tags):
        super(DummyMatrixBase, self).__init__(backend, ioshape, iopacking, tags)

        if initval is not None:
            self.data = initval
        else:
            self.data = np.empty((self.nrow, self.ncol))

    def _get(self):
        return self.data

    def _set(self, buf):
        self.data = buf

    @property
    def nbytes(self):
        return self.data.nbytes


class DummyMatrix(DummyMatrixBase, base.Matrix):
    pass


class DummyConstMatrix(DummyMatrixBase, base.ConstMatrix):
    def __init__(self, backend, initval, iopacking, tags):
        ioshape = initval.shape
        super(DummyConstMatrix, self).__init__(backend, ioshape, initval,
                                               iopacking, tags)


class DummySparseMatrix(DummyMatrixBase, base.SparseMatrix):
    def __init__(self, backend, initval, iopacking, tags):
        ioshape = initval.shape
        super(DummySparseMatrix, self).__init__(backend, ioshape, initval,
                                                iopacking, tags)


class DummyMPIMatrix(DummyMatrix, base.MPIMatrix):
    pass


class DummyMatrixBank(base.MatrixBank):
    def __init__(self, mats, tags):
        super(DummyMatrixBank, self).__init__(mats)


class DummyView(base.View):
    def __init__(self, matmap, rcmap, tags):
        pass


class DummyMPIView(base.MPIView):
    def __init__(self, matmap, rcmap, tags):
        pass


class DummyKernel(base.Kernel):
    pass


class DummyQueue(base.Queue):
    def __lshift__(self, iterable):
        pass

    def __mod__(self, iterable):
        pass

class DummyBackend(base.Backend):
    def _matrix(self, *args, **kwargs):
        return DummyMatrix(self, *args, **kwargs)

    def _matrix_bank(self, *args, **kwargs):
        return DummyMatrixBank(self, *args, **kwargs)

    def _mpi_matrix(self, *args, **kwargs):
        return DummyMPIMatrix(self, *args, **kwargs)

    def _const_matrix(self, *args, **kwargs):
        return DummyConstMatrix(self, *args, **kwargs)

    def _sparse_matrix(self, *args, **kwargs):
        return DummySparseMatrix(self, *args, **kwargs)

    def _is_sparse(self, mat, tags):
        if 'sparse' in tags:
            return True
        elif np.count_nonzero(mat) < 0.33*mat.size:
            return True
        else:
            return False

    def _view(self, *args, **kwargs):
        return DummyView(*args, **kwargs)

    def _mpi_view(self, *args, **kwargs):
        return DummyMPIView(*args, **kwargs)

    def _kernel(self, name, *args, **kwargs):
        validateattr = '_validate_' + name

        # Check the kernel name
        if not hasattr(self, validateattr):
            raise PyFRInvalidKernelError("'{}' is not a valid kernel"\
                                         .format(name))

        # Call the validator method to check the arguments
        getattr(self, validateattr)(*args, **kwargs)

        return DummyKernel()

    def _queue(self, *args, **kwargs):
        return DummyQueue(*args, **kwargs)

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
