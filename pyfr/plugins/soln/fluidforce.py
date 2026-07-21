import numpy as np

from pyfr.cache import memoize
from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.plugins.mixins import BackendMixin, PublishMixin, SeriesWriterMixin
from pyfr.plugins.soln.base import BaseSolnPlugin
from pyfr.quadrules.surface import SurfaceIntegrator


class FluidForceIntegrator(SurfaceIntegrator):
    def __init__(self, cfg, cfgsect, system, bcname, morigin):
        con = system.mesh.bcon.get(bcname)

        super().__init__(cfg, cfgsect, system.ele_map, con, flags='s')

        if self.locs and morigin is not None:
            self.rfpts = {k: loc - morigin[:, None, None]
                          for k, loc in self.locs.items()}


class FluidForcePlugin(PublishMixin, SeriesWriterMixin, BackendMixin,
                       BaseSolnPlugin):
    name = 'fluidforce'
    systems = 'euler|navier-stokes'
    dimensions = '2|3'

    def __init__(self, intg, cfgsect, suffix):
        super().__init__(intg, cfgsect, suffix)

        comm, rank, root = get_comm_rank_root()

        # Check if we need to compute viscous force
        self._viscous = 'navier-stokes' in intg.system.name

        # Viscous correction
        self._viscorr = self.cfg.get('solver', 'viscosity-correction', 'none')

        # Constant variables
        self._constants = self.cfg.items_as('constants', float)

        # Moments
        morigin = None
        mcomp = 3 if self.ndims == 3 else 1
        self._mcomp = mcomp if self.cfg.hasopt(cfgsect, 'morigin') else 0
        if self._mcomp:
            morigin = np.array(self.cfg.getliteral(cfgsect, 'morigin'))
            if len(morigin) != self.ndims:
                raise ValueError(f'morigin must have {self.ndims} components')

        # Force coefficients (optional; require reference conditions)
        self._cnames, self._cdirs, self._qref = [], [], None
        if self.cfg.hasopt(cfgsect, 'area-ref'):
            # Reference conditions: the plugin section overrides [constants]
            rhoinf = self._ref_cond('rho-inf')
            uinf = self._ref_cond('u-inf')

            # Reference dynamic pressure times the reference area
            aref = self.cfg.getfloat(cfgsect, 'area-ref')
            self._qref = 0.5*rhoinf*uinf**2*aref
            self._cnames, self._cdirs = self._force_dirs()

        # Moment coefficients additionally need a reference length
        self._mcnames, self._mcdirs, self._mqref = [], [], None
        want_moms = (self._qref is not None and self._mcomp and
                     self.cfg.hasopt(cfgsect, 'len-ref'))
        if want_moms:
            self._mqref = self._qref*self.cfg.getfloat(cfgsect, 'len-ref')

            # 2D moment: scalar (z); 3D: roll/pitch/yaw = drag/side/lift
            if self.ndims == 2:
                self._mcnames, self._mcdirs = ['cm'], [np.array([1.0])]
            else:
                drag, lift, side = self._cdirs
                self._mcnames = ['cmr', 'cmp', 'cmy']
                self._mcdirs = [drag, side, lift]

        # See which ranks have the boundary
        bcranks = comm.gather(suffix in intg.system.mesh.bcon, root=root)

        # The root rank needs to open the output file
        if rank == root:
            if not any(bcranks):
                raise RuntimeError(f'Boundary {suffix} does not exist')

            self._init_series(intg, self._fields)

        # Set interpolation matrices and quadrature weights
        self.ff_int = FluidForceIntegrator(self.cfg, cfgsect, intg.system,
                                           suffix, morigin)

        # Initialise backend infrastructure
        self._init_backend(intg)

        # Number of output components per element
        ncomp = self.ndims + self._mcomp
        self._nout = (2 if self._viscous else 1)*ncomp

        # Initialise GPU kernel infrastructure
        self._init_kernels(intg)

    def _ref_cond(self, name):
        if self.cfg.hasopt(self.cfgsect, name):
            return self.cfg.getfloat(self.cfgsect, name)
        elif name in self._constants:
            return self._constants[name]
        else:
            raise ValueError(f'Force coefficients require {name} in '
                             '[constants] or the plugin section')

    def _force_dirs(self):
        cfg, cfgsect, ndims = self.cfg, self.cfgsect, self.ndims

        def unit(name):
            d = np.array(cfg.getliteral(cfgsect, name), dtype=float)
            if len(d) != ndims:
                raise ValueError(f'{name} must have {ndims} components')
            return d / np.linalg.norm(d)

        drag = unit('drag-dir')

        # Lift defaults to drag rotated by +90 degrees in 2D
        if ndims == 2 and not cfg.hasopt(cfgsect, 'lift-dir'):
            lift = np.array([-drag[1], drag[0]])
        else:
            lift = unit('lift-dir')

        # Drag and lift must be orthogonal
        if abs(drag @ lift) > 1e-6:
            raise ValueError('Force coefficient directions must be '
                             'orthogonal')

        if ndims == 2:
            names, dirs = ['cd', 'cl'], [drag, lift]
        else:
            # Side = lift x drag, so (drag, side, lift) is right-handed
            side = np.cross(lift, drag)
            side /= np.linalg.norm(side)

            names, dirs = ['cd', 'cl', 'cs'], [drag, lift, side]

        return names, dirs

    @property
    def _fields(self):
        fields = ['px', 'py', 'pz'][:self.ndims]
        if self._mcomp:
            fields += ['mpx', 'mpy', 'mpz'][3 - self._mcomp:]
        if self._viscous:
            fields += ['vx', 'vy', 'vz'][:self.ndims]
            if self._mcomp:
                fields += ['mvx', 'mvy', 'mvz'][3 - self._mcomp:]

        return fields + self._cnames + self._mcnames

    def _init_kernels(self, intg):
        backend = self.backend

        # Register our kernel template
        backend.pointwise.register('pyfr.plugins.soln.kernels.fluidforce')

        # Precompute per-face data and upload to device
        fi = self.ff_int
        self._efaces = []

        for (etype, fidx), m0 in fi.m0.items():
            nfpts, nupts = m0.shape
            eidxs = fi.eidxs[etype, fidx]
            neles = len(eidxs)

            # Weighted normals: qwts * norms → (nfpts, ndims, neles)
            qwts, norms = fi.qwts[etype, fidx], fi.norms[etype, fidx]
            wnorms = (qwts[None, :, None]*norms).transpose(1, 0, 2)

            # Moment arm positions at face points
            if self._mcomp:
                rfpts = fi.rfpts[etype, fidx].transpose(1, 0, 2)
                rfpts_mat = backend.const_matrix(rfpts, tags={'align'})
            else:
                rfpts_mat = None

            self._efaces.append({
                'idx': self._etype_map[etype], 'eidxs': eidxs,
                'nupts': nupts, 'nfpts': nfpts,
                'm0': backend.const_matrix(m0, tags={'align'}),
                'wnorms': backend.const_matrix(wnorms, tags={'align'}),
                'rfpts': rfpts_mat,
                'pf': backend.matrix((self._nout, neles), tags={'align'})
            })

    @memoize
    def _get_kerns(self, uidx):
        kerns = []

        for ef in self._efaces:
            eidxs = ef['eidxs']
            nupts, nfpts = ef['nupts'], ef['nfpts']

            tplargs = {
                'ndims': self.ndims, 'nvars': self.nvars, 'nupts': nupts,
                'nfpts': nfpts, 'nout': self._nout, 'viscous': self._viscous,
                'visc_corr': self._viscorr, 'mcomp': self._mcomp,
                'c': self._constants
            }

            # Solution view into scal_upts[uidx]
            u = self._make_view(self._ele_banks[ef['idx']][uidx], eidxs,
                                (nupts, self.nvars))

            # Gradient view (bank-independent)
            if self._viscous:
                gradu = self._make_view(self._grad_banks[ef['idx']], eidxs,
                                        (self.ndims*nupts, self.nvars))
            else:
                gradu = None

            kerns.append(self.backend.pointwise.fluidforce(
                tplargs=tplargs, dims=[len(eidxs)], u=u, gradu=gradu,
                m0=ef['m0'], wnorms=ef['wnorms'], rfpts=ef['rfpts'],
                pf=ef['pf']
            ))

        return kerns

    def __call__(self, intg):
        # MPI info
        comm, rank, root = get_comm_rank_root()

        # Compute gradients on device if viscous
        if self._viscous:
            intg.compute_grads()

        # Launch all force kernels
        self.backend.run_kernels(self._get_kerns(intg.idxcurr))

        # Collect per-element results and sum
        fm = np.zeros(self._nout)

        for ef in self._efaces:
            fm += ef['pf'].get().sum(axis=-1)

        fm = fm.reshape(-1, self.ndims + self._mcomp)

        # Reduce and output if we're the root rank
        if rank != root:
            comm.Reduce(fm, None, op=mpi.SUM, root=root)
        else:
            comm.Reduce(mpi.IN_PLACE, fm, op=mpi.SUM, root=root)

            # Force components (pressure, then viscous) for publishing
            pn = ['px', 'py', 'pz'][:self.ndims]
            pvals = dict(zip(pn, fm[0]))
            if self._viscous:
                vn = ['vx', 'vy', 'vz'][:self.ndims]
                pvals.update(zip(vn, fm[-1]))

            # Coefficients from the total (pressure + viscous) force/moment
            if self._qref is not None:
                ftot = fm[:, :self.ndims].sum(axis=0)
                coeffs = [ftot @ d / self._qref for d in self._cdirs]

                if self._mcnames:
                    mtot = fm[:, self.ndims:].sum(axis=0)
                    coeffs += [mtot @ d / self._mqref for d in self._mcdirs]

                pvals.update(zip(self._cnames + self._mcnames, coeffs))
                samps = np.append(fm.ravel(), coeffs)
            else:
                samps = fm

            self._write(intg.tcurr, samps)
            self._publish(intg, **pvals)

    def trigger_write(self, intg):
        self(intg)
