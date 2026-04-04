import re

import numpy as np

from pyfr.cache import memoize
from pyfr.inifile import NoOptionError
from pyfr.quadrules import get_quadrule
from pyfr.regions import ConstructiveRegion
from pyfr.util import first


def compile_expr(expr, privars, ndims):
    # Map primitive variables, gradients, and coordinates to C arrays
    subs = {v: f'pri[{i}]' for i, v in enumerate(privars)}
    for i, v in enumerate(privars):
        for j, d in enumerate('xyz'[:ndims]):
            subs[f'grad_{v}_{d}'] = f'grad_pri[{i}][{j}]'
    for d, c in enumerate('xyz'[:ndims]):
        subs[c] = f'ploc[{d}]'

    p = '|'.join(re.escape(k) for k in subs)
    return re.sub(rf'\b({p})\b', lambda m: subs[m[1]], expr)


class BackendFieldReducer:
    def __init__(self, backend, cfg, cfgsect, intg, exprs, reduceop):
        self._backend = backend
        self._cfg = cfg
        self._cfgsect = cfgsect
        self._reduceop = reduceop
        self._nexprs = len(exprs)

        efrst = first(intg.system.ele_map.values())
        privars = efrst.privars
        self._nvars = efrst.nvars
        self._ndims = efrst.ndims

        # Compile expressions
        cexprs = [compile_expr(e, privars, self._ndims) for e in exprs]

        # Detect if expressions use gradients or coordinates
        allexprs = ' '.join(exprs)
        self._has_grads = bool(re.search(r'\bgrad_', allexprs))
        self._has_ploc = bool(re.search(r'\b[xyz]\b', allexprs))

        # Region mask callback
        region = cfg.get(cfgsect, 'region', '*')
        if region == '*':
            self._make_mask = None
        else:
            crgn = ConstructiveRegion(region)
            self._make_mask = lambda pts: crgn.pts_in_region(
                np.moveaxis(pts, 1, 2)
            )

        # Register the fieldeval kernel
        backend.pointwise.register('pyfr.plugins.soln.kernels.fieldeval')

        # Element banks and gradient banks
        self._ele_banks = intg.system.ele_banks
        self._grad_banks = intg.system.eles_vect_upts

        # Shared template arguments
        self._tplargs = {
            'ndims': self._ndims, 'nvars': self._nvars,
            'nexprs': self._nexprs, 'exprs': cexprs,
            'reduceop': reduceop, 'c': cfg.items_as('constants', float),
            'has_grads': self._has_grads, 'use_views': False,
            'eos_mod': efrst.eos_kernel_module,
        }

        # Per-element-type kernel data
        self._edata = [self._prepare_etype(k, *v)
                       for k, v in enumerate(intg.system.ele_map.items())]

    def _get_quad_interp(self, ename, eles):
        cfg, cfgsect = self._cfg, self._cfgsect
        rname = cfg.get(f'solver-elements-{ename}', 'soln-pts')

        try:
            qrule = cfg.get(cfgsect, f'quad-pts-{ename}', rname)
            try:
                qdeg = cfg.getint(cfgsect, f'quad-deg-{ename}')
            except NoOptionError:
                qdeg = cfg.getint(cfgsect, 'quad-deg')

            r = get_quadrule(ename, qrule, qdeg=qdeg)
        except NoOptionError:
            r = get_quadrule(ename, rname, eles.nupts)

        m0 = eles.basis.ubasis.nodal_basis_at(r.pts)

        # Skip interpolation if m0 is identity
        if (m0.shape[0] == m0.shape[1] and
            np.allclose(m0, np.eye(m0.shape[0]))):
            return r, None
        else:
            return r, m0

    def _get_wts_mask(self, eles, r):
        fpdtype = self._backend.fpdtype
        rcpdjacs = eles.rcpdjac_at_np(r.pts)

        # Quadrature weights
        wts_np = r.wts[:, None] / rcpdjacs

        # Region mask: (nqpts, neles)
        mk = self._make_mask
        if mk is not None and not np.all(inside := mk(eles.ploc_at_np(r.pts))):
            mask_np = inside.astype(fpdtype)
        else:
            mask_np = None

        # Total volume (masked if applicable)
        w = wts_np * mask_np if mask_np is not None else wts_np
        wts_total = np.sum(w)

        return wts_np, mask_np, wts_total

    def _alloc_etype_mats(self, eles, nqpts, m0, wts_np, mask_np, ploc_np):
        backend = self._backend
        fpdtype = backend.fpdtype
        cmat = lambda a: backend.const_matrix(a, tags={'align'})
        mat = lambda s: backend.matrix(s, dtype=fpdtype, tags={'align'})

        neles = eles.neles
        has_mask = mask_np is not None

        # Reuse the existing backend ploc matrix when at upts
        ploc_mat = None
        if self._has_ploc:
            if m0 is None:
                ploc_mat = eles.ploc_at('upts')
            else:
                ploc_mat = cmat(ploc_np.astype(fpdtype))

        # Weights matrix
        wts_mat = None
        if self._reduceop == 'sum':
            if has_mask:
                wts_mat = cmat((wts_np * mask_np).astype(fpdtype))
            else:
                wts_mat = cmat(wts_np.astype(fpdtype))
        elif has_mask:
            wts_mat = cmat(mask_np)

        out_mat = mat((self._nexprs, neles))

        # Interpolation matrix and temp buffers
        if m0 is not None:
            m0_const = cmat(m0.astype(fpdtype))
            u_temp = mat((nqpts, self._nvars, neles))
            gshape = (self._ndims, nqpts, self._nvars, neles)
            gradu_temp = mat(gshape) if self._has_grads else None
        else:
            m0_const = u_temp = gradu_temp = None

        tplargs = {**self._tplargs, 'has_wts': wts_mat is not None}

        return dict(
            nqpts=nqpts, neles=neles, nupts=eles.nupts, m0_const=m0_const,
            u_temp=u_temp, gradu_temp=gradu_temp, ploc_mat=ploc_mat,
            wts_mat=wts_mat, out_mat=out_mat, tplargs=tplargs,
        )

    def _prepare_etype(self, eidx, ename, eles):
        r, m0 = self._get_quad_interp(ename, eles)
        wts_np, mask_np, wts_total = self._get_wts_mask(eles, r)
        ploc_np = eles.ploc_at_np(r.pts)
        mats = self._alloc_etype_mats(eles, len(r.pts), m0, wts_np, mask_np,
                                      ploc_np)

        return {**mats, 'eidx': eidx, 'wts_total': wts_total}

    def total_volume(self):
        return sum(d['wts_total'] for d in self._edata)

    @memoize
    def _get_kerns(self, uidx):
        backend = self._backend
        kerns = []

        for d in self._edata:
            eidx = d['eidx']

            # Solution data
            soln = self._ele_banks[eidx][uidx]
            gradu = self._grad_banks[eidx] if self._has_grads else None

            # Interpolation kernels
            if (m0 := d['m0_const']) is not None:
                kerns.append(backend.kernel('mul', m0, soln, out=d['u_temp']))
                u = d['u_temp']

                if gradu is not None and d['gradu_temp'] is not None:
                    nupts, nqpts = d['nupts'], d['nqpts']
                    gt = d['gradu_temp']
                    for dd in range(self._ndims):
                        gs = gradu.slice(dd*nupts, (dd + 1)*nupts)
                        go = gt.slice(dd*nqpts, (dd + 1)*nqpts)
                        kerns.append(backend.kernel('mul', m0, gs, out=go))
                    gradu = gt
            else:
                u = soln

            # Pointwise fieldeval kernel
            kwargs = dict(u=u, out=d['out_mat'])

            if self._has_grads and gradu is not None:
                kwargs['gradu'] = gradu
            if d['ploc_mat'] is not None:
                kwargs['ploc'] = d['ploc_mat']
            if d['wts_mat'] is not None:
                kwargs['wts'] = d['wts_mat']

            kerns.append(backend.pointwise.fieldeval(
                tplargs=d['tplargs'], dims=[d['nqpts'], d['neles']],
                **kwargs
            ))

        return kerns

    def __call__(self, intg):
        if self._has_grads:
            intg.compute_grads()

        kerns = self._get_kerns(intg.idxcurr)
        for kern in kerns:
            if hasattr(kern, 'bind'):
                kern.bind(t=intg.tcurr)
        self._backend.run_kernels(kerns)

        # Fetch results from device and reduce across element types
        ident = {'sum': 0.0, 'min': np.inf, 'max': -np.inf}
        result = np.full(self._nexprs, ident[self._reduceop])

        for d in self._edata:
            out = d['out_mat'].get()

            match self._reduceop:
                case 'sum':
                    result += out.sum(axis=1)
                case 'min':
                    result = np.minimum(result, out.min(axis=1))
                case 'max':
                    result = np.maximum(result, out.max(axis=1))

        return result
