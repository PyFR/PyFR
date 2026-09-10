import numpy as np

from pyfr.integrators.base import (BaseIntegrator, _common_plugin_prop,
                                   kernel_getter)
from pyfr.integrators.implicit.precond import Preconditioner
from pyfr.integrators.implicit.tolerance import get_linear_tol_controller
from pyfr.integrators.registers import DynamicScalarRegister
from pyfr.mpiutil import get_comm_rank_root, mpi, scal_coll
from pyfr.nputil import bfloat16, is_bf16
from pyfr.progress import format_s
from pyfr.util import subclass_where


class BaseImplicitIntegrator(BaseIntegrator):
    formulation = 'implicit'

    # If a linear solver is required or not
    requires_linear_solver = False

    # Scratch space for a preconditioner
    _precond_temp = DynamicScalarRegister()

    _pcdtype_map = {
        'double': np.float64,
        'single': np.float32,
        'half': np.float16,
        'bf16': bfloat16,
    }

    def __init__(self, backend, systemcls, mesh, initsoln, cfg):
        super().__init__(backend, mesh, initsoln, cfg)

        sect = 'solver-time-integrator'

        # Sanity checks
        if self.controller_needs_errest and not self.stepper_has_errest:
            raise TypeError('Incompatible stepper/controller combination')

        if cfg.get('solver', 'shock-capturing', 'none') == 'entropy-filter':
            raise TypeError('Entropy filtering not compatible with '
                            'implicit time stepping')

        # Finite-difference perturbation for JFNK and preconditioner
        if (fdeps := cfg.get(sect, 'fd-eps', 'auto')) == 'auto':
            self._fd_eps = backend.fpdtype_eps**0.5
            self._fd_eps_adapt = True
        else:
            self._fd_eps = float(fdeps)
            self._fd_eps_adapt = False

        # Obtain the preconditioner class, if any
        pname = cfg.get(sect, 'precond', 'none').lower()
        self._pccls = subclass_where(Preconditioner, name=pname)

        # Preconditioner working precision
        pstr = cfg.get(sect, 'precond-precision', 'double').lower()
        try:
            self._pcdtype = self._pcdtype_map[pstr]
        except KeyError:
            raise ValueError('Invalid precond-precision: must be double, '
                             'single, half, or bf16')

        # Clamp preconditioner dtype to backend dtype if necessary
        if (not is_bf16(self._pcdtype) and
            np.finfo(self._pcdtype).bits > np.finfo(backend.fpdtype).bits):
            self._pcdtype = backend.fpdtype

        # Allocate preconditioner scratch only when active
        self._size_register(self._precond_temp, self._pccls.active)

        # Construct the relevant system
        self.system = systemcls(backend, mesh, initsoln, self._registers, cfg,
                                self.serialiser,
                                needs_cfl=self.controller_needs_cfl)

        # Assign register numbers to our member variables
        self._assign_registers()

        # Event handlers for advance_to
        self.plugins = self._get_plugins(initsoln)

        # Hook for subclasses to modify extents before commit
        self._pre_commit()

        # Commit the system
        self.system.commit()

        # Hook for subclasses which require committed storage
        self._post_commit()

        # Index of the register number containing the solution
        self.idxcurr = 0

        # Global degree of freedom count
        self.gndofs = self._get_gndofs()

        # Adaptive controller for the inner linear (Krylov) tolerance
        self._tol_controller = get_linear_tol_controller(cfg, self.serialiser,
                                                         initsoln)

    def _pre_commit(self):
        ext = self.backend.get_extent('solve')

        self._preconditioner = self._pccls(
            self.backend, self.system, ext, fd_eps=self._fd_eps,
            pcdtype=self._pcdtype
        )

    def _post_commit(self):
        self._preconditioner.init_kernels()

    def collect_stats(self, stats):
        super().collect_stats(stats)

        pc = self._preconditioner
        if pc.active:
            stats.set('solver-time-integrator', 'precond-build-time',
                      pc.build_wtime_total)
            stats.set('solver-time-integrator', 'precond-nbuilds', pc.nbuilds)

    def _setup_progress(self):
        super()._setup_progress()

        if self._preconditioner.active:
            def precond():
                pc = self._preconditioner
                if pc.computed:
                    return (f'Precond {format_s(pc.build_wtime)} '
                            f'@ t = {pc.build_t:.2f}')
                else:
                    return None

            self.progress.add_status_field(precond, hold=3.0)
            self._preconditioner.progress = self.progress

    def _compute_fd_eps(self, u_reg):
        # If we are adaptive then recompute eps
        if self._fd_eps_adapt:
            unorm = self._norm2(u_reg)
            self._fd_eps = ((1 + unorm)*self.backend.fpdtype_eps)**0.5

    @property
    def _precond_computed(self):
        return self._preconditioner.computed

    @property
    def _precond_gdt_built(self):
        return self._preconditioner.gdt_built

    @property
    def _precond_build_wtime(self):
        return self._preconditioner.build_wtime

    def _invalidate_precond(self):
        self._preconditioner.invalidate()

    def _compute_precond(self, t, u_reg, gamma_dt, rhs_fn, f0_reg, up_reg,
                         eps_scales=()):
        self._preconditioner.construct(
            t, u_reg, gamma_dt, rhs_fn, f0_reg, up_reg, self._add,
            self._precond_temp, eps_scales=eps_scales
        )

    def _apply_precond(self, in_reg, out_reg, in_scale=(), out_scale=()):
        kerns = self._get_precond_kerns(in_reg, out_reg,
                                        in_scale=tuple(in_scale),
                                        out_scale=tuple(out_scale))
        self.backend.run_kernels(kerns)

    @kernel_getter
    def _get_precond_kerns(self, emats, in_reg, out_reg, *, in_scale,
                           out_scale=()):
        idx = self.system.ele_banks.index(emats)
        return self._preconditioner.apply_kernel(emats, idx, in_reg, out_reg,
                                                 in_scale, out_scale)

    @_common_plugin_prop('_curr_soln')
    def soln(self):
        return self.system.ele_scal_upts(self.idxcurr)

    @_common_plugin_prop('_curr_grad_soln')
    def grad_soln(self):
        self.compute_grads()
        return [e.get() for e in self.system.eles_vect_upts]

    @_common_plugin_prop('_curr_dt_soln')
    def dt_soln(self):
        soln = [np.require(s, requirements='O') for s in self.soln]

        self.system.rhs(self.tcurr, self.idxcurr, self.idxcurr)
        dt_soln = [np.require(s, requirements='O')
                   for s in self.system.ele_scal_upts(self.idxcurr)]

        # Reset current register with original contents
        for e, s in zip(self.system.ele_banks, soln):
            e[self.idxcurr].set(s)

        return dt_soln

    def _norm2(self, r, *, weights=(), norm_gndofs=False):
        comm, rank, root = get_comm_rank_root()

        # Run the kernels
        kerns = self._get_norm2_kerns(r, weights=tuple(weights))
        self.backend.run_kernels(kerns, wait=True)

        # Reduce over element types and ranks
        norm = scal_coll(comm.Allreduce, sum(k.retval[0] for k in kerns))
        scale = self.gndofs if norm_gndofs else 1

        return (norm / scale)**0.5

    @kernel_getter
    def _get_norm2_kerns(self, emats, x, *, weights=()):
        expr = ['(x/w)*(x/w)'] if weights else ['x*x']
        pvars = {'w': weights} if weights else {}
        return self.backend.kernel('reduction', 'sum', expr,
                                   {'x': emats[x]}, pvars=pvars)

    def _dot(self, a, b):
        return float(self._multidot(a, b)[0])

    def _multidot(self, a, b0, *bn):
        comm, rank, root = get_comm_rank_root()
        kerns = self._get_multidot_kerns(a, b0, *bn)

        self.backend.run_kernels(kerns, wait=True)

        # Reduce over element types and ranks and return
        results = sum([k.retval.astype(float) for k in kerns])
        comm.Allreduce(mpi.IN_PLACE, results)
        return results

    @kernel_getter
    def _get_multidot_kerns(self, emats, a, *bn):
        exprs = [f'a*b{i}' for i in range(len(bn))]
        vvars = {f'b{i}': emats[b] for i, b in enumerate(bn)}
        vvars['a'] = emats[a]
        return self.backend.kernel('reduction', 'sum', exprs, vvars)
