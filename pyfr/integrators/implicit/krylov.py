
from pyfr.cache import memoize
from pyfr.integrators.base import kernel_getter
from pyfr.integrators.implicit.base import BaseImplicitIntegrator
from pyfr.integrators.registers import ScalarRegister
from pyfr.util import ndrange


class BaseKrylovSolver(BaseImplicitIntegrator):
    krylov_name = None
    _precond_temp = ScalarRegister()

    def __init__(self, backend, systemcls, mesh, initsoln, cfg):
        sect = 'solver-time-integrator'

        # Finite difference perturbation for JFNK and preconditioner
        if cfg.hasopt(sect, 'krylov-eps'):
            self._krylov_eps = cfg.getfloat(sect, 'krylov-eps')
        else:
            self._krylov_eps = backend.fpdtype_eps**0.5

        # Preconditioner
        self._precond = cfg.get(sect, 'krylov-precond', 'block-jacobi').lower()
        if self._precond not in ('none', 'block-jacobi'):
            raise ValueError('Invalid preconditioner: must be none or '
                             'block-jacobi')

        super().__init__(backend, systemcls, mesh, initsoln, cfg)

        # Precision-dependent breakdown tolerance
        self._breakdown_tol = 1e3*self.backend.fpdtype_eps

        if self._precond == 'block-jacobi':
            self._init_precond()

    def _init_precond(self):
        for kname in ['applyprecond', 'precondperturb',
                      'precondextract', 'precondscale']:
            self.backend.pointwise.register(
                f'pyfr.integrators.implicit.kernels.{kname}'
            )

        # Upload colour arrays to backend
        ixdtype = self.backend.ixdtype
        self._precond_colours = []
        for et in self.system.ele_types:
            colours = self.system.mesh.colours[et].astype(ixdtype)
            neles = len(colours)
            self._precond_colours.append(
                self.backend.matrix((1, neles), colours.reshape(1, neles),
                                    tags={'align'}, dtype=ixdtype)
            )

        # Allocate Jacobian blocks (after inversion, holds preconditioner)
        self._precond_J_blocks = []
        for et in self.system.ele_types:
            nupts, nvars, neles = self.system.ele_shapes[et]
            bsize = nupts * nvars
            self._precond_J_blocks.append(self.backend.matrix(
                (bsize, bsize, neles), tags={'align'}
            ))

        self._precond_computed = False

    @memoize
    def _get_precond_perturb_extract_kerns(self, u_reg, up_reg, f0_reg, eps,
                                            eps_scales):
        ele_banks = self.system.ele_banks
        colours, J_blocks = self._precond_colours, self._precond_J_blocks
        kerns = []
        for eb, c, J in zip(ele_banks, colours, J_blocks):
            nupts, nvars, neles = eb[u_reg].ioshape
            bsize = nupts * nvars
            tplargs = {'nupts': nupts, 'nvars': nvars, 'bsize': bsize,
                       'eps_scales': eps_scales}

            perturb_kern = self.backend.kernel(
                'precondperturb', tplargs=tplargs, dims=[neles],
                u=eb[u_reg], up=eb[up_reg], colours=c,
                pcolour='pcolour', pcidx='pcidx',
                colour='colour', cidx='cidx', eps=eps
            )
            extract_kern = self.backend.kernel(
                'precondextract', tplargs=tplargs, dims=[neles],
                f=eb[self._precond_temp], f0=eb[f0_reg], J=J,
                colours=c, colour='colour', cidx='cidx', eps=eps
            )
            kerns.append((nupts, perturb_kern, extract_kern))

        return kerns

    @memoize
    def _get_precond_scale_inv_kerns(self, u_reg):
        ele_banks, J_blocks = self.system.ele_banks, self._precond_J_blocks

        scale_kerns, inv_kerns = [], []
        for eb, J in zip(ele_banks, J_blocks):
            nupts, nvars, neles = eb[u_reg].ioshape
            bsize = nupts * nvars
            scale_kerns.append(self.backend.kernel(
                'precondscale', tplargs={'bsize': bsize}, dims=[neles],
                J=J, gamma='gamma'
            ))
            inv_kerns.append(self.backend.kernel('batched_inv', m=J))

        return scale_kerns, inv_kerns

    def _compute_precond(self, t, u_reg, gamma_dt, rhs_fn, f0_reg, up_reg,
                         eps_scales=()):
        if self._precond == 'none' or self._precond_computed:
            return

        eps = self._krylov_eps
        nvars = self.system.nvars

        # Default to uniform scaling if not provided
        eps_scales = eps_scales or (1,)*nvars

        # Find max nupts and number of colours
        max_nupts = max(s[0] for s in self.system.ele_shapes.values())
        ncolours = max(c.max() for c in self.system.mesh.colours.values()) + 1

        # Copy u to up (up will be perturbed, u stays untouched)
        self._add(0, up_reg, 1, u_reg)

        kern_info = self._get_precond_perturb_extract_kerns(u_reg, up_reg,
                                                            f0_reg, eps,
                                                            eps_scales)

        # Previous (colour, cidx) per element type for mixed meshes
        prev = [(-1, -1) for _ in kern_info]

        for colour in range(ncolours):
            for i, k in ndrange(max_nupts, nvars):
                cidx = i*nvars + k

                perturb_kerns, extract_kerns = [], []
                for idx, (nupts, pk, ek) in enumerate(kern_info):
                    if i < nupts:
                        pk.bind(pcolour=prev[idx][0], pcidx=prev[idx][1],
                                colour=colour, cidx=cidx)
                        ek.bind(colour=colour, cidx=cidx)
                        perturb_kerns.append(pk)
                        extract_kerns.append(ek)
                        prev[idx] = (colour, cidx)

                self.backend.run_kernels(perturb_kerns)
                rhs_fn(t, up_reg, self._precond_temp)
                self.backend.run_kernels(extract_kerns)

        # Get scale/inv kernels (cached) and run
        scale_kerns, inv_kerns = self._get_precond_scale_inv_kerns(u_reg)

        # Bind gamma and run scale kernels
        for kern in scale_kerns:
            kern.bind(gamma=gamma_dt)

        self.backend.run_kernels(scale_kerns + inv_kerns)

        self._precond_computed = True

    def _apply_precond(self, in_reg, out_reg, in_scale=(), out_scale=()):
        kerns = self._get_precond_kerns(in_reg, out_reg,
                                        in_scale=tuple(in_scale),
                                        out_scale=tuple(out_scale))
        self.backend.run_kernels(kerns)

    @kernel_getter
    def _get_precond_kerns(self, emats, in_reg, out_reg, *, in_scale,
                           out_scale=()):
        idx = self.system.ele_banks.index(emats)
        nupts, nvars, neles = emats[in_reg].ioshape

        tplargs = {'nupts': nupts, 'nvars': nvars, 'block_size': nupts*nvars,
                   'in_scale': in_scale, 'out_scale': out_scale}

        return self.backend.kernel(
            'applyprecond', tplargs=tplargs, dims=[neles],
            x=emats[in_reg], minv=self._precond_J_blocks[idx], y=emats[out_reg]
        )
