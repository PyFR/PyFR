import re

from pyfr.util import first


class BaseBlasExtKernels:
    pvar_idx = None

    def axnpby(self, *arr, in_scale=(), in_scale_idxs=(), out_scale=()):
        if any(arr[0].traits != x.traits for x in arr[1:]):
            raise ValueError('Incompatible matrix types')

        nv = len(arr)
        ncola = arr[0].ioshape[-2]

        tplargs = dict(ncola=ncola, nv=nv, in_scale_idxs=in_scale_idxs,
                       in_scale=in_scale, out_scale=out_scale)

        return self._axnpby(arr, tplargs)


    def reduction(self, rop, expr, vvars, svars=[], pvars={}):
        # Ensure all matrices are compatible
        fvvar = first(vvars.values())
        if any(v.traits != fvvar.traits for v in vvars.values()):
            raise ValueError('Incompatible matrix types')

        # Index each expression so var => var[idx]
        vnames = '|'.join(map(re.escape, vvars))
        exprs = [re.sub(rf'\b({vnames})\b', r'\1[idx]', e) for e in expr]

        # Substitute pvars with indexed access if pvar_idx is set
        if pvars and self.pvar_idx:
            pnames = '|'.join(map(re.escape, pvars))
            exprs = [re.sub(rf'\b({pnames})\b', rf'_pv_\1[{self.pvar_idx}]', e)
                     for e in exprs]

        # Common template arguments
        init_val = 0 if rop == 'sum' else -self.backend.fpdtype_max
        tplargs = dict(exprs=exprs, nexprs=len(exprs), init_val=init_val,
                       rop=rop, svars=svars, vvars=list(vvars), pvars=pvars)

        return self._reduction(fvvar, vvars, svars, tplargs)
