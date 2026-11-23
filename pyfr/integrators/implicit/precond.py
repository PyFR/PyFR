import math
import time

import numpy as np

from pyfr.cache import memoize
from pyfr.integrators.implicit.luinv import LUInverse
from pyfr.mpiutil import get_comm_rank_root, mpi, scal_coll
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

        # Allocate the per-element inverse and Jacobian staging matrices
        self.inv = backend.tiled_matrix(block_shape=(n, n), nmats=self.neles,
                                        dtype=pcdtype)
        self.jstage = backend.matrix((n, n, self.nemax), dtype=wdtype,
                                     extent=extent, tags={'align'})

        # Allocate the workspace of the inversion beside the staging matrix
        self.luinv = LUInverse(backend, self.jstage, self.inv, self.ceidxs,
                               extent)

    def _gather_view(self, mat, eidxs):
        # Pad the gather map to capacity by replicating its final entry
        e = np.pad(eidxs, (0, self.nemax - len(eidxs)), mode='edge')

        vmap = self.backend.view_mapping(np.full(self.nemax, mat.mid),
                                         np.zeros(self.nemax, dtype=int), e,
                                         np.ones(self.nemax, dtype=int),
                                         vshape=(self.nupts, self.nvars))

        return self.backend.view(vmap)

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

    def apply_kern(self, x, y, in_scale, out_scale):
        tplargs = {'nupts': self.nupts, 'nvars': self.nvars, 'n': self.n,
                   'pcdtype': npdtype_to_ctype(self.inv.dtype),
                   'in_scale': bool(in_scale), 'out_scale': bool(out_scale)}

        # Upload the scale factors of the variables which are given
        scales = {}
        for name, scale in [('isc', in_scale), ('osc', out_scale)]:
            if scale:
                sc = np.array([scale], dtype=x.dtype)
                scales[name] = self.backend.const_matrix(sc)

        return self.backend.kernel(
            'precondapply', tplargs=tplargs, dims=[self.neles],
            minv=self.inv, x=x, y=y, **scales
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
        self.luinv.invert(colour, scale)


class Preconditioner:
    active = True
    computed = True
    gdt_built = 0.0
    build_wtime = 0.0
    build_wtime_total = 0.0
    build_t = 0.0
    nbuilds = 0

    # Decades of gamma*dt drift from the build which force a rebuild
    _gdt_shift = 0.5

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
        backend.pointwise.register(
            'pyfr.integrators.implicit.kernels.precondapply'
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
        # Which element types contain each colour
        self._centries = [[e for e in self._eles if c in e.ceidxs]
                          for c in range(self.ncolours)]

    def invalidate(self):
        self.computed = False

    def construct(self, t, u_reg, gamma_dt, rhs_fn, f0_reg, up_reg, add_fn,
                  ptmp, eps_scales=()):
        # Rebuild once gamma*dt has moved well away from the build
        if self.computed and self.gdt_built:
            if abs(math.log10(gamma_dt / self.gdt_built)) > self._gdt_shift:
                self.invalidate()

        if self.computed:
            return

        comm, _, _ = get_comm_rank_root()
        t0 = time.perf_counter() - self.backend.kernel_setup_wtime

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

        # Agree on the build time across ranks
        wtime = time.perf_counter() - self.backend.kernel_setup_wtime - t0
        wtime = scal_coll(comm.Allreduce, wtime, op=mpi.MAX)

        self.computed = True
        self.gdt_built = gamma_dt
        self.build_t = t
        self.build_wtime = wtime
        self.build_wtime_total += wtime
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
                for e in ents:
                    e.invert(colour, gamma_dt)

    def apply_kernel(self, emats, etidx, in_reg, out_reg, in_scale=(),
                     out_scale=()):
        e = self._eles[etidx]

        return e.apply_kern(emats[in_reg], emats[out_reg], in_scale,
                            out_scale)
