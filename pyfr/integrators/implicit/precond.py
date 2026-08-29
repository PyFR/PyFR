from collections import namedtuple
import time

import numpy as np

from pyfr.cache import memoize
from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.nputil import npdtype_to_ctype
from pyfr.progress import NullProgressBar
from pyfr.util import ndrange


_ElementInfo = namedtuple('_ElementInfo', 'etidx nupts neles n tile')


class Preconditioner:
    active = True
    computed = True
    gdt_built = 0.0
    build_wtime = 0.0
    build_wtime_total = 0.0
    build_t = 0.0
    nbuilds = 0

    def __init__(self, backend, system, ext, *, fd_eps=0, pcdtype=None):
        self.backend = backend
        self.system = system

        # Progress bar; replaced by the integrator when running interactively
        self.progress = NullProgressBar()

    def init_kernels(self):
        pass

    def construct(self, t, u_reg, gamma_dt, rhs_fn, f0_reg, up_reg, add_fn,
                  ptmp, eps_scales=()):
        pass

    def invalidate(self):
        pass


class NullPreconditioner(Preconditioner):
    name = 'none'
    active = False

    def apply_kernel(self, *args, **kwargs):
        pass


class BlockJacobiPreconditioner(Preconditioner):
    name = 'block-jacobi'

    def __init__(self, backend, system, ext, *, fd_eps, pcdtype):
        super().__init__(backend, system, ext, fd_eps=fd_eps,
                         pcdtype=pcdtype)
        self._fd_eps = fd_eps
        self._pcdtype = pcdtype
        self._wdtype = np.float32 if pcdtype == np.float16 else pcdtype
        self.computed = False

        # Build per-element-type info (block sizes, tile sizes)
        self._einfos = einfos = []
        for etidx, etype in enumerate(system.ele_types):
            nupts, _, neles = system.ele_shapes[etype]
            n = nupts*system.nvars
            tile = backend.optimal_tile_shape(n, pcdtype)
            einfos.append(_ElementInfo(etidx, nupts, neles, n, tile))

        self._nupts = [info.nupts for info in einfos]

        # Register pointwise kernel templates
        backend.pointwise.register(
            'pyfr.integrators.implicit.kernels.precondperturb'
        )
        backend.pointwise.register(
            'pyfr.integrators.implicit.kernels.precondextract'
        )

        # Upload colour arrays and element index lists
        self._init_colours()

        # Compute global max nupts and ncolours across all MPI ranks
        comm, _, _ = get_comm_rank_root()
        self.nvars = system.nvars
        self.max_nupts = comm.allreduce(max(self._nupts), op=mpi.MAX)
        self.ncolours = comm.allreduce(
            max(max(ce) + 1 for ce in self._ceidxs), op=mpi.MAX
        )

        # Allocate the per-element inverse and per-colour Jacobian matrices
        groups = [ext.group() for _ in range(self.ncolours)]

        self._invs, self._jstages = [], []
        for etidx, _, neles, n, tile in self._einfos:
            inv = backend.tiled_matrix(block_size=n, nmats=neles,
                                       tile_shape=tile, dtype=self._pcdtype)
            jstages = {}
            for colour, eidxs in self._ceidxs[etidx].items():
                jstages[colour] = backend.matrix(
                    (n, n, eidxs.ioshape[-1]), dtype=self._wdtype,
                    extent=groups[colour], tags={'align'}
                )

            self._invs.append(inv)
            self._jstages.append(jstages)

    def _init_colours(self):
        self._colours, self._ceidxs, self._ceidxs_h = [], [], []
        for colours in self.system.mesh.colours.values():
            # Upload the colour array for this element type
            self._colours.append(self.backend.matrix(
                (1, len(colours)), colours[None, :], tags={'align'},
                dtype=self.backend.ixdtype
            ))

            # Build per-colour element index lists
            ceidxs, ceidxs_h = {}, {}
            for colour in map(int, np.unique(colours)):
                eidxs = np.flatnonzero(colours == colour)
                ceidxs[colour] = self.backend.const_matrix(
                    eidxs[None, :], dtype=self.backend.ixdtype,
                    tags={'align'}
                )
                ceidxs_h[colour] = eidxs

            self._ceidxs.append(ceidxs)
            self._ceidxs_h.append(ceidxs_h)

    def _make_view(self, mat, eidxs, vshape):
        n = len(eidxs)
        return self.backend.view(np.full(n, mat.mid), np.zeros(n, dtype=int),
                                 eidxs, np.ones(n, dtype=int), vshape=vshape)

    def init_kernels(self):
        backend = self.backend
        ekerns = []
        items = zip(self._einfos, self._invs, self._jstages)
        for (etidx, nupts, *_), inv, jstages in items:
            ckerns = {}
            for colour, jstage in jstages.items():
                ikern = backend.kernel('batched_inv_tiled', m=jstage, out=inv,
                                       eidxs=self._ceidxs[etidx][colour])
                ckerns[colour] = (etidx, nupts, ikern)

            ekerns.append(ckerns)

        self._ckerns = [[ek[c] for ek in ekerns if c in ek]
                        for c in range(self.ncolours)]

    def invalidate(self):
        self.computed = False

    def construct(self, t, u_reg, gamma_dt, rhs_fn, f0_reg, up_reg, add_fn,
                  ptmp, eps_scales=()):
        if self.computed:
            return

        t0 = time.perf_counter()

        # Default to uniform scaling if not provided
        self._escales = eps_scales or (1,)*self.nvars
        self._eeps = tuple(self._fd_eps*s for s in self._escales)

        # Copy u to up (up will be perturbed, u stays untouched)
        add_fn(0, up_reg, 1, u_reg)

        # Obtain the perturbation and extraction kernels
        pkerns = self._get_pkerns(u_reg, up_reg, self._escales)
        ext_kerns = self._get_ekerns(f0_reg, ptmp)

        # Build the block Jacobian via finite differences
        self._construct(t, gamma_dt, pkerns, ext_kerns, rhs_fn, up_reg, ptmp)

        self.computed = True
        self.gdt_built = gamma_dt
        self.build_t = t
        self.build_wtime = time.perf_counter() - t0
        self.build_wtime_total += self.build_wtime
        self.nbuilds += 1

    def _construct(self, t, gamma_dt, pkerns, ext_kerns, rhs_fn, up_reg, ptmp):
        prev = [(-1, -1)]*len(pkerns)

        total = self.ncolours*self.max_nupts*self.nvars
        with self.progress.task('Precond', total) as task:
            for colour, entries in enumerate(self._ckerns):
                for upt, var in ndrange(self.max_nupts, self.nvars):
                    cidx = upt*self.nvars + var

                    # Perturb up and evaluate the RHS
                    kerns, idxs = self._bind_pkerns(pkerns, prev, colour, cidx,
                                                    upt)
                    self.backend.run_kernels(kerns)
                    rhs_fn(t, up_reg, ptmp)

                    # Note what colours and columns were perturbed
                    for idx in idxs:
                        prev[idx] = (colour, cidx)

                    # Extract the Jacobian column into scratch
                    ekerns = []
                    for etidx, nupts, _ in entries:
                        if upt < nupts:
                            ek = ext_kerns[etidx][colour]
                            ek.bind(cidx=cidx, inv_eps=1 / self._eeps[var])
                            ekerns.append(ek)
                    self.backend.run_kernels(ekerns)

                    task.advance()

                # Scale and invert this colour's contribution
                for _, _, ikern in entries:
                    ikern.bind(scale=gamma_dt)
                self.backend.run_kernels([ikern for _, _, ikern in entries])

    def _bind_pkerns(self, pkerns, prev, colour, cidx, upt):
        kerns, idxs = [], []
        for idx, (nupts, pkern) in enumerate(zip(self._nupts, pkerns)):
            if upt < nupts:
                pkern.bind(pcolour=prev[idx][0], pcidx=prev[idx][1],
                           colour=colour, cidx=cidx)
                kerns.append(pkern)
                idxs.append(idx)

        return kerns, idxs

    @memoize
    def _get_pkerns(self, u_reg, up_reg, escales):
        kerns = []

        items = zip(self._einfos, self.system.ele_banks, self._colours)
        for info, ele_banks, colours in items:
            tplargs = {'nupts': info.nupts, 'nvars': self.nvars, 'n': info.n,
                       'eps_scales': escales}
            pkern = self.backend.kernel(
                'precondperturb', tplargs=tplargs, dims=[info.neles],
                u=ele_banks[u_reg], up=ele_banks[up_reg], colours=colours,
                eps=self._fd_eps
            )
            kerns.append(pkern)

        return kerns

    @memoize
    def _get_ekerns(self, f0_reg, ptmp):
        ekerns = []

        for info, jstages in zip(self._einfos, self._jstages):
            ele_banks = self.system.ele_banks[info.etidx]
            tplargs = {'nupts': info.nupts, 'nvars': self.nvars, 'n': info.n,
                       'wkdtype': npdtype_to_ctype(self._wdtype)}
            ckerns = {}

            for colour, jstage in jstages.items():
                eidxs_h = self._ceidxs_h[info.etidx][colour]
                cneles = len(eidxs_h)

                vshape = (info.nupts, self.nvars)
                fv = self._make_view(ele_banks[ptmp], eidxs_h, vshape)
                f0v = self._make_view(ele_banks[f0_reg], eidxs_h, vshape)

                ckerns[colour] = self.backend.kernel(
                    'precondextract', tplargs=tplargs, dims=[cneles],
                    f=fv, f0=f0v, jac=jstage
                )

            ekerns.append(ckerns)

        return ekerns

    def apply_kernel(self, emats, etidx, in_reg, out_reg, in_scale=(),
                     out_scale=()):
        x = emats[in_reg]
        y = emats[out_reg]
        info = self._einfos[etidx]
        minv = self._invs[etidx]

        return self.backend.kernel(
            'batched_tiled_matvec', x=x, minv=minv, y=y, nupts=info.nupts,
            nvars=self.nvars, in_scale=in_scale, out_scale=out_scale
        )
