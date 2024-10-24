import numpy as np

from pyfr.cache import memoize
from pyfr.integrators.dual.pseudo.base import BaseDualPseudoIntegrator


def _get_coefficients_from_txt(scheme):
    coeffs_mat = []

    for i, l in enumerate(scheme.splitlines(), start=1):
        a = [float(f) for f in l.split()]

        if len(a) == i:
            coeffs_mat.append(a)
        else:
            raise ValueError('Invalid coefficient structure in scheme')

    # The last row is the b vector
    return coeffs_mat[:-1], coeffs_mat[-1]


class BaseDualPseudoStepper(BaseDualPseudoIntegrator):
    @property
    def ntotiters(self):
        return self.npseudosteps

    def collect_stats(self, stats):
        # Total number of RHS evaluations
        stats.set('solver-time-integrator', 'nfevals',
                  self.pseudo_stepper_nfevals)

        # Total number of pseudo-steps
        stats.set('solver-time-integrator', 'npseudosteps', self.npseudosteps)

    def _rhs_with_dts(self, t, uin, fout):
        # Compute -∇·f
        self.system.rhs(t, uin, fout)

        # Registers and coefficients
        vals = [self.stepper_coeffs[-1], -1/self._dt, 1]
        regs = [fout, self._idxcurr, self._source_regidx]

        # Physical stepper source addition -∇·f - dQ/dt
        self._addv(vals, regs, subdims=self._subdims)


class DualEulerPseudoStepper(BaseDualPseudoStepper):
    pseudo_stepper_name = 'euler'
    pseudo_stepper_order = 1
    pseudo_stepper_nregs = 2
    pseudo_stepper_has_lerrest = False

    @property
    def pseudo_stepper_nfevals(self):
        return self.npseudosteps

    def step(self, t):
        self.npseudosteps += 1

        add = self._add
        rhs = self._rhs_with_dts

        r0, r1 = self._pseudo_stepper_regidx

        if r0 != self._idxcurr:
            r0, r1 = r1, r0

        rhs(t, r0, r1)
        add(0, r1, 1, r0, self._dtau, r1)

        return r1, r0


class DualTVDRK3PseudoStepper(BaseDualPseudoStepper):
    pseudo_stepper_name = 'tvd-rk3'
    pseudo_stepper_order = 3
    pseudo_stepper_nregs = 3
    pseudo_stepper_has_lerrest = False

    @property
    def pseudo_stepper_nfevals(self):
        return 3*self.npseudosteps

    def step(self, t):
        self.npseudosteps += 1

        add = self._add
        rhs = self._rhs_with_dts
        dtau = self._dtau

        # Get the bank indices for pseudo-registers (n+1,m; n+1,m+1; rhs),
        # where m = pseudo-time and n = real-time
        r0, r1, r2 = self._pseudo_stepper_regidx

        # Ensure r0 references the bank containing u(n+1,m)
        if r0 != self._idxcurr:
            r0, r1 = r1, r0

        # First stage;
        # r2 = -∇·f(r0) - dQ/dt; r1 = r0 + dtau*r2
        rhs(t, r0, r2)
        add(0, r1, 1, r0, dtau, r2)

        # Second stage;
        # r2 = -∇·f(r1) - dQ/dt; r1 = 3/4*r0 + 1/4*r1 + 1/4*dtau*r2
        rhs(t, r1, r2)
        add(1/4, r1, 3/4, r0, dtau/4, r2)

        # Third stage;
        # r2 = -∇·f(r1) - dQ/dt; r1 = 1/3*r0 + 2/3*r1 + 2/3*dtau*r2
        rhs(t, r1, r2)
        add(2/3, r1, 1/3, r0, 2*dtau/3, r2)

        # Return the index of the bank containing u(n+1,m+1)
        return r1, r0


class DualRK4PseudoStepper(BaseDualPseudoStepper):
    pseudo_stepper_name = 'rk4'
    pseudo_stepper_order = 4
    pseudo_stepper_nregs = 3
    pseudo_stepper_has_lerrest = False

    @property
    def pseudo_stepper_nfevals(self):
        return 4*self.npseudosteps

    def step(self, t):
        self.npseudosteps += 1

        add = self._add
        rhs = self._rhs_with_dts
        dtau = self._dtau

        # Get the bank indices for pseudo-registers (n+1,m; n+1,m+1; rhs),
        # where m = pseudo-time and n = real-time
        r0, r1, r2 = self._pseudo_stepper_regidx

        # Ensure r0 references the bank containing u(n+1,m)
        if r0 != self._idxcurr:
            r0, r1 = r1, r0

        # First stage; r1 = -∇·f(r0) - dQ/dt;
        rhs(t, r0, r1)

        # Second stage; r2 = r0 + dtau/2*r1; r2 = -∇·f(r2) - dQ/dt;
        add(0, r2, 1, r0, dtau/2, r1)
        rhs(t, r2, r2)

        # As no subsequent stages depend on the first stage we can
        # reuse its register to start accumulating the solution with
        # r1 = r0 + dtau/6*r1 + dtau/3*r2
        add(dtau/6, r1, 1, r0, dtau/3, r2)

        # Third stage; here we reuse the r2 register
        # r2 = r0 + dtau/2*r2 - dtau/2*dQ/dt
        # r2 = -∇·f(r2) - dQ/dt;
        add(dtau/2, r2, 1, r0)
        rhs(t, r2, r2)

        # Accumulate; r1 = r1 + dtau/3*r2
        add(1, r1, dtau/3, r2)

        # Fourth stage; again we reuse r2
        # r2 = r0 + dtau*r2
        # r2 = -∇·f(r2) - dQ/dt;
        add(dtau, r2, 1, r0)
        rhs(t, r2, r2)

        # Final accumulation r1 = r1 + dtau/6*r2 = u(n+1,m+1)
        add(1, r1, dtau/6, r2)

        # Return the index of the bank containing u(n+1,m+1)
        return r1, r0


class DualEmbeddedPairPseudoStepper(BaseDualPseudoStepper):
    # Coefficients
    a = []
    b = []
    bhat = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Compute the error coeffs
        self.e = [b - bh for b, bh in zip(self.b, self.bhat)]

        self._nstages = len(self.b)

        # Allocate storage for the local pseudo time-step field
        self.dtau_upts = [self.backend.matrix(shape, np.ones(shape)*self._dtau,
                                              tags={'align'})
                          for shape in self.system.ele_shapes.values()]

        # Register a pointwise kernel for the low-storage stepper
        self.backend.pointwise.register(
            'pyfr.integrators.dual.pseudo.kernels.rkvdh2pseudo'
        )

    @memoize
    def _get_rkvdh2pseudo_kerns(self, stage, r1, r2, rold, rerr=None):
        kerns = []
        tplargs = {'a': self.a, 'b': self.b, 'e': self.e, 'stage': stage,
                   'nstages': self._nstages, 'nvars': self.system.nvars,
                   'errest': rerr is not None}

        for dims, em, dtaum in zip(self.system.ele_shapes.values(),
                                   self.system.ele_banks, self.dtau_upts):
            kern = self.backend.kernel(
                'rkvdh2pseudo', tplargs=tplargs, dims=[dims[0], dims[2]],
                dtau=dtaum, r1=em[r1], r2=em[r2], rold=em[rold],
                rerr=em[rerr] if rerr else None,
            )
            kerns.append(kern)

        return kerns

    @property
    def pseudo_stepper_has_lerrest(self):
        return self.pseudo_controller_needs_lerrest and self.bhat


class DualRKVdH2RPseudoStepper(DualEmbeddedPairPseudoStepper):
    @property
    def pseudo_stepper_nfevals(self):
        return len(self.b)*self.npseudosteps

    @property
    def pseudo_stepper_nregs(self):
        return 4 if self.pseudo_stepper_has_lerrest else 3

    def step(self, t):
        self.npseudosteps += 1

        run_kernels, rhs = self.backend.run_kernels, self._rhs_with_dts

        rold = self._idxcurr
        r1, r2, *rerr = set(self._pseudo_stepper_regidx) - {rold}

        # Evaluate the stages in the scheme
        for i in range(self._nstages):
            # Compute -∇·f - dQ/dt
            rhs(t, r2 if i > 0 else rold, r2)

            # Fetch the appropriate RK accumulation kernels
            kerns = self._get_rkvdh2pseudo_kerns(i, r1, r2, rold, *rerr)

            # Execute
            run_kernels(kerns)

            # Swap
            r1, r2 = r2, r1

        # Return
        return (r2, rold, *rerr)


class DualRK34PseudoStepper(DualRKVdH2RPseudoStepper):
    pseudo_stepper_name = 'rk34'
    pseudo_stepper_order = 3

    a = [
        11847461282814 / 36547543011857,
        3943225443063 / 7078155732230,
        -346793006927 / 4029903576067
    ]

    b = [
        1017324711453 / 9774461848756,
        8237718856693 / 13685301971492,
        57731312506979 / 19404895981398,
        -101169746363290 / 37734290219643
    ]

    bhat = [
        15763415370699 / 46270243929542,
        514528521746 / 5659431552419,
        27030193851939 / 9429696342944,
        -69544964788955 / 30262026368149
    ]


class DualRK45PseudoStepper(DualRKVdH2RPseudoStepper):
    pseudo_stepper_name = 'rk45'
    pseudo_stepper_order = 4

    a = [
        970286171893 / 4311952581923,
        6584761158862 / 12103376702013,
        2251764453980 / 15575788980749,
        26877169314380 / 34165994151039
    ]

    b = [
        1153189308089 / 22510343858157,
        1772645290293 / 4653164025191,
        -1672844663538 / 4480602732383,
        2114624349019 / 3568978502595,
        5198255086312 / 14908931495163
    ]

    bhat = [
        1016888040809 / 7410784769900,
        11231460423587 / 58533540763752,
        -1563879915014 / 6823010717585,
        606302364029 / 971179775848,
        1097981568119 / 3980877426909
    ]
