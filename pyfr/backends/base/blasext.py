import re

import numpy as np

from pyfr.util import first


class BaseBlasExtKernels:
    # Dimensions of the kernel which performs a reduction
    reduction_ndim = 1

    def __init__(self, backend):
        super().__init__(backend)

        backend.pointwise.register('pyfr.backends.base.kernels.reduce')

    def reduction(self, rop, expr, vvars, svars=[], pvars={}):
        backend = self.backend

        # Ensure all matrices are compatible
        fvvar = first(vvars.values())
        if any(v.traits != fvvar.traits for v in vvars.values()):
            raise ValueError('Incompatible matrix types')

        # Find the dimensions of the operands
        ncdim = len(fvvar.ioshape) >= 3
        ncola = fvvar.ioshape[-2] if ncdim else 1
        nrow, ncolb = fvvar.traits[1], fvvar.ioshape[-1]

        # Walk the rows within each lane or as the rows of a 2D kernel
        ndim = self.reduction_ndim
        if ndim == 1:
            dims, rix = [ncolb], '[_i]'
        else:
            dims, rix = [nrow, ncolb], ''

        # Index each expression so var => var[_i][k] with pvars as literals
        ixs = [f'{rix}[{k}]' if ncdim else rix for k in range(ncola)]
        exprs = [[self._index_expr(e, k, ix, vvars, pvars)
                  for k, ix in enumerate(ixs)] for e in expr]

        # Initialise the result with the identity of the operation
        nexprs = len(exprs)
        ident = 0 if rop == 'sum' else -backend.fpdtype_max
        init = backend.const_matrix(np.full((1, nexprs), ident),
                                    tags={'noblock'})
        out = backend.matrix((1, nexprs), tags={'noblock'})

        # Combine the expressions over every row and column of the operands
        tplargs = {
            'exprs': exprs, 'nexprs': nexprs, 'rop': rop, 'ndim': ndim,
            'ncdim': ncdim, 'nrow': nrow, 'ncola': ncola,
            'ident': repr(float(ident)), 'vvars': list(vvars), 'svars': svars
        }
        pw = backend.pointwise.reduce(tplargs, dims=dims, _r=out, **vvars)

        class ReductionKernel(backend.ordered_meta_kernel_cls):
            @property
            def retval(self):
                return out.get()[0]

            if svars:
                def bind(self, *consts):
                    pw.bind(**dict(zip(svars, consts)))

        return ReductionKernel([backend.kernel('copy', out, init), pw])

    def _index_expr(self, e, k, ix, vvars, pvars):
        # Index the matrices of an expression and inline the column's pvars
        vnames = '|'.join(map(re.escape, vvars))
        s = re.sub(rf'\b({vnames})\b', rf'\1{ix}', e)
        if pvars:
            pnames = '|'.join(map(re.escape, pvars))
            s = re.sub(rf'\b({pnames})\b',
                       lambda m: repr(float(pvars[m[1]][k])), s)

        return s
