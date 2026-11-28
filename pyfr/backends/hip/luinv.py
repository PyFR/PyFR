from functools import cached_property
from pathlib import Path

import numpy as np

from pyfr.backends.hip.provider import HIPKernel
from pyfr.integrators.implicit.luinv import LUInverse
from pyfr.nputil import npdtype_to_ctype, npdtype_to_ctypestype


# The entry file of the kernels and the headers it includes
SRCDIR = Path(__file__).parent / 'kernels' / 'luinv'

# C types of the storage types
CTYPES = {'double': 'double', 'float': 'float', 'half': '_Float16',
          'bf16': 'luinv::bf16'}


class HIPLUKernel(HIPKernel):
    def __init__(self, fun, grid, block):
        super().__init__()

        self.fun = fun
        self.params = fun.make_params(grid, block)

    def set_args(self, *args):
        self.params.set_args(*args)

    def run(self, stream):
        self.fun.exec_async(stream, self.params)


class HIPLUInverse(LUInverse):
    """The LU inverse as plain HIP kernels with MFMA.
    
        Runs three separate kernels: build, factor, and solve.
    """

    @staticmethod
    def supported(backend):
        # Wave64 GPUs with the fp64 and fp32 16x16x4 MFMA
        arch = backend.props['gcn_arch_name']
        return (backend.subgrpsz == 64 and
                arch.startswith(('gfx90a', 'gfx94', 'gfx95')))

    def _tplargs(self, m, out):
        t = super()._tplargs(m, out)

        # Pad to whole strips of 16 columns; the workspace holds the packed
        # factor, then the inverses of the diagonal blocks of U, as
        # config.hpp has it
        t['npad'] = npad = -(-t['n'] // 16)*16
        t['wsz'] = npad*(npad + 16)

        # Factorise more matrices at once, within a workspace of 2**28
        # entries (2 GiB in fp64), starting each part at a whole AoSoA group of the matrices
        nwork = max(512, min(4096, 2**28 // t['wsz']))
        t['nwork'] = min(nwork // 64 * 64, m.ioshape[2])

        return t

    @cached_property
    def module(self):
        b, t, out = self.backend, self.tplargs, self.out
        ncu = b.props['multiprocessor_count']

        defs = {
            'LUINV_N': t['n'], 'LUINV_FTYPE': t['ftype'],
            'LUINV_STYPE': CTYPES[t['stype']],
            'LUINV_OTYPE': CTYPES[t['otype']],
            'LUINV_SCALE_T': npdtype_to_ctype(b.fpdtype),
            'LUINV_IXTYPE': {4: 'int', 8: 'int64_t'}[
                np.dtype(b.ixdtype).itemsize
            ],
            'LUINV_TR': out.trows, 'LUINV_TC': out.tcols,
            'LUINV_PADC': out.padc, 'LUINV_NDIES': max(1, b.ndies),
            'LUINV_FEW': int(t['nwork'] <= 3*ncu)
        }

        flags = [f'--gpu-architecture={b.props["gcn_arch_name"]}',
                 '-munsafe-fp-atomics', f'-I{SRCDIR}',
                 *(f'-D{k}={v}' for k, v in defs.items())]

        with b.region('compile'):
            src = (SRCDIR / 'lu_inv.hip').read_text()
            code = b.compiler.build('lu_inv', src, flags)

            return b.hip.load_module(code)

    def _function(self, name, argtypes):
        argtypes = [npdtype_to_ctypestype(t) for t in argtypes]
        return self.module.get_function(name, argtypes)

    @cached_property
    def kerns(self):
        b, t = self.backend, self.tplargs
        ix, ptr, nb = b.ixdtype, np.uintp, t['npad'] // 16

        with b.region('kernel-setup'):
            build = self._function('lu_inv_build',
                                   [ix, b.fpdtype, ptr, ix, ptr, ptr])
            factor = self._function('lu_inv_factor', [ix] + [ptr]*3)
            solve = self._function('lu_inv_solve', [ix] + [ptr]*5)

        # The launch bounds of the kernels give their waves per block
        k1w, wpb = factor.max_threads // 64, solve.max_threads // 64

        # The kernels for each number of items inverted at once
        kerns = {}
        for s in {v.n for ps in self.parts.values() for _, v in ps}:
            kerns[s] = [
                HIPLUKernel(build, (-(-s // 64)*t['npad'], 1, 1),
                            (256, 1, 1)),
                HIPLUKernel(factor, (s, 1, 1), (64*k1w, 1, 1)),
                HIPLUKernel(solve, (s*-(-nb // wpb), 1, 1), (64*wpb, 1, 1))
            ]

        return kerns

    def invert(self, key, scale):
        for src, inv in self.parts[key]:
            kb, kf, ks = self.kerns[inv.n]
            kb.set_args(inv.n, scale, *src.kargs('ml'), self.wk, self.ws)
            kf.set_args(inv.n, self.wk, self.ws, self.perm)
            ks.set_args(inv.n, *inv.kargs('v'), self.perm, self.wk, self.ws)

            self.backend.run_kernels([kb, kf, ks])
