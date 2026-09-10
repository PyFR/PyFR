import time

import numpy as np

from pyfr.cache import memoize
from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.nputil import npdtype_to_ctype
from pyfr.progress import NullProgressBar
from pyfr.util import first, ndrange


class _ElementPrecond:
    def __init__(self, system, etidx, pcdtype, wdtype, extent):
        self.backend = backend = system.backend
        self.system = system
        self.etidx = etidx
        self.wdtype = wdtype

        etype = system.ele_types[etidx]
        self.nvars = nvars = system.nvars
        self.nupts, _, self.neles = system.ele_shapes[etype]
        self.n = n = self.nupts*nvars

        # Upload the colour array for this element type
        colours = system.mesh.colours[etype]
        self.cmat = backend.matrix((1, len(colours)), colours[None, :],
                                   tags={'align'}, dtype=backend.ixdtype)

        # Build per-colour element index vectors
        self.ceidxs = {int(c): np.flatnonzero(colours == c)
                       for c in np.unique(colours)}
        self.nemax = max(map(len, self.ceidxs.values()))

        # Upload the per-colour element index vectors
        self.eidxs = {c: backend.const_matrix(e[None, :], tags={'align'},
                                              dtype=backend.ixdtype)
                      for c, e in self.ceidxs.items()}

        # Allocate the per-element inverse and Jacobian staging matrices
        tile = backend.optimal_tile_shape(n, pcdtype)
        self.inv = backend.tiled_matrix(block_size=n, nmats=self.neles,
                                        tile_shape=tile, dtype=pcdtype)
        self.jstage = backend.matrix((n, n, self.nemax), dtype=wdtype,
                                     extent=extent, tags={'align'})

    def init_ikern(self):
        # Build the inversion kernel at staging capacity
        self.ikern = self.backend.kernel('batched_inv_tiled', m=self.jstage,
                                         out=self.inv,
                                         eidxs=first(self.eidxs.values()))

    def _gather_view(self, mat, eidxs):
        # Pad the gather map to capacity by replicating its final entry
        e = np.pad(eidxs, (0, self.nemax - len(eidxs)), mode='edge')

        return self.backend.view(np.full(self.nemax, mat.mid),
                                 np.zeros(self.nemax, dtype=int), e,
                                 np.ones(self.nemax, dtype=int),
                                 vshape=(self.nupts, self.nvars))

    @memoize
    def _colour_views(self, reg):
        # Capacity-width gather windows onto this register per colour
        mat = self.system.ele_banks[self.etidx][reg]
        return {c: self._gather_view(mat, e)
                for c, e in self.ceidxs.items()}

    @memoize
    def _perturb_kern(self, escales):
        bank = self.system.ele_banks[self.etidx][0]
        tplargs = {'nupts': self.nupts, 'nvars': self.nvars, 'n': self.n,
                   'eps_scales': escales}

        # Build against the first bank; prepare binds the live operands
        return self.backend.kernel(
            'precondperturb', tplargs=tplargs, dims=[self.neles],
            u=bank, up=bank, colours=self.cmat, eps=0
        )

    @memoize
    def _extract_kern(self):
        tplargs = {'nupts': self.nupts, 'nvars': self.nvars, 'n': self.n,
                   'wkdtype': npdtype_to_ctype(self.wdtype)}

        # Build against the first bank; bind_colour binds the live views
        v = first(self._colour_views(0).values())
        return self.backend.kernel(
            'precondextract', tplargs=tplargs, dims=[self.nemax],
            jac=self.jstage, f=v, f0=v
        )

    def prepare(self, u_reg, up_reg, f0_reg, ptmp, eps, escales):
        banks = self.system.ele_banks[self.etidx]
        self.prev = (-1, -1)

        # Fetch the kernels and bind the current operands
        self.pkern = self._perturb_kern(escales)
        self.pkern.bind(u=banks[u_reg], up=banks[up_reg], eps=eps)
        self.ekern = self._extract_kern()

        # Per-colour gather windows onto the residual registers
        self._fviews = self._colour_views(ptmp)
        self._f0views = self._colour_views(f0_reg)

    def bind_colour(self, colour):
        # Bind this colour's gather windows to the extract kernel
        self.ekern.bind(f=self._fviews[colour], f0=self._f0views[colour])

    def perturb(self, colour, cidx):
        # Bind the previous and current perturbation columns
        self.pkern.bind(pcolour=self.prev[0], pcidx=self.prev[1],
                        colour=colour, cidx=cidx)
        self.prev = (colour, cidx)

        return self.pkern

    def extract(self, cidx, inv_eps):
        self.ekern.bind(cidx=cidx, inv_eps=inv_eps)

        return self.ekern

    def invert(self, colour, scale):
        self.ikern.bind(scale=scale, eidxs=self.eidxs[colour])

        return self.ikern


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
        self._wdtype = np.float32 if pcdtype == np.float16 else pcdtype
        self.computed = False
        self.nvars = system.nvars

        # Register pointwise kernel templates
        backend.pointwise.register(
            'pyfr.integrators.implicit.kernels.precondperturb'
        )
        backend.pointwise.register(
            'pyfr.integrators.implicit.kernels.precondextract'
        )

        # Build the preconditioner state for each element type
        extent = ext.struct()
        self._eles = [
            _ElementPrecond(system, i, pcdtype, self._wdtype, extent)
            for i in range(len(system.ele_types))
        ]

        # Compute global max nupts and ncolours across all MPI ranks
        comm, _, _ = get_comm_rank_root()
        self.max_nupts = comm.allreduce(max(e.nupts for e in self._eles),
                                        op=mpi.MAX)
        self.ncolours = comm.allreduce(
            max(max(e.ceidxs) + 1 for e in self._eles), op=mpi.MAX
        )

    def init_kernels(self):
        # Build the inversion kernel for each element type
        for e in self._eles:
            e.init_ikern()

        # Which element types contain each colour
        self._centries = [[e for e in self._eles if c in e.ceidxs]
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

        # Prepare the perturbation and extraction kernels
        for e in self._eles:
            e.prepare(u_reg, up_reg, f0_reg, ptmp, self._fd_eps,
                      self._escales)

        # Build the block Jacobian via finite differences
        self._construct(t, gamma_dt, rhs_fn, up_reg, ptmp)

        self.computed = True
        self.gdt_built = gamma_dt
        self.build_t = t
        self.build_wtime = time.perf_counter() - t0
        self.build_wtime_total += self.build_wtime
        self.nbuilds += 1

    def _construct(self, t, gamma_dt, rhs_fn, up_reg, ptmp):
        total = self.ncolours*self.max_nupts*self.nvars
        with self.progress.task('Precond', total) as task:
            for colour, ents in enumerate(self._centries):
                # Bind this colour's gather windows to the extract kernels
                for e in ents:
                    e.bind_colour(colour)

                for upt, var in ndrange(self.max_nupts, self.nvars):
                    cidx = upt*self.nvars + var

                    # Perturb up and evaluate the RHS
                    pkerns = [e.perturb(colour, cidx)
                              for e in self._eles if upt < e.nupts]
                    self.backend.run_kernels(pkerns)
                    rhs_fn(t, up_reg, ptmp)

                    # Extract the Jacobian column into scratch
                    ekerns = [e.extract(cidx, 1 / self._eeps[var])
                              for e in ents if upt < e.nupts]
                    self.backend.run_kernels(ekerns)

                    task.advance()

                # Scale and invert this colour's contribution
                ikerns = [e.invert(colour, gamma_dt) for e in ents]
                self.backend.run_kernels(ikerns)

    def apply_kernel(self, emats, etidx, in_reg, out_reg, in_scale=(),
                     out_scale=()):
        e = self._eles[etidx]

        return self.backend.kernel(
            'batched_tiled_matvec', x=emats[in_reg], minv=e.inv,
            y=emats[out_reg], nupts=e.nupts, nvars=self.nvars,
            in_scale=in_scale, out_scale=out_scale
        )
