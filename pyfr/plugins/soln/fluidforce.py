import numpy as np

from pyfr.cache import memoize
from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.plugins.common import DatasetAppender, init_csv, open_hdf5_a
from pyfr.plugins.mixins import BackendMixin, PublishMixin
from pyfr.plugins.soln.base import BaseSolnPlugin
from pyfr.quadrules.surface import SurfaceIntegrator


class FluidForceIntegrator(SurfaceIntegrator):
    def __init__(self, cfg, cfgsect, system, bcname, morigin):
        con = system.mesh.bcon.get(bcname)

        super().__init__(cfg, cfgsect, system.ele_map, con, flags='s')

        if self.locs and morigin is not None:
            self.rfpts = {k: loc - morigin for k, loc in self.locs.items()}


class FluidForcePlugin(PublishMixin, BackendMixin, BaseSolnPlugin):
    name = 'fluidforce'
    systems = ['euler', 'navier-stokes']
    dimensions = [2, 3]

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

        # See which ranks have the boundary
        bcranks = comm.gather(suffix in intg.system.mesh.bcon, root=root)

        # The root rank needs to open the output file
        if rank == root:
            if not any(bcranks):
                raise RuntimeError(f'Boundary {suffix} does not exist')

            match self.cfg.get(cfgsect, 'file-format', 'csv'):
                case 'csv':
                    self._init_csv()
                case 'hdf5':
                    self._init_hdf5()
                case _:
                    raise ValueError('Invalid file format')

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

    @property
    def _header(self):
        header = ['t', 'px', 'py', 'pz'][:self.ndims + 1]
        if self._mcomp:
            header += ['mpx', 'mpy', 'mpz'][3 - self._mcomp:]
        if self._viscous:
            header += ['vx', 'vy', 'vz'][:self.ndims]
            if self._mcomp:
                header += ['mvx', 'mvy', 'mvz'][3 - self._mcomp:]

        return ','.join(header)

    def _init_csv(self):
        self.csv = init_csv(self.cfg, self.cfgsect, self._header, nflush=1)
        self._write = self._write_csv

    def _write_csv(self, t, forces):
        self.csv(t, *forces.ravel())

    def _init_hdf5(self):
        outf = open_hdf5_a(self.cfg.get(self.cfgsect, 'file'))
        nvars = 1 + (2 if self._viscous else 1)*(self.ndims + self._mcomp)

        dset = self.cfg.get(self.cfgsect, 'file-dataset')
        if dset in outf:
            ff = outf[dset]

            if ff.shape[1] != nvars:
                raise ValueError('Invalid dataset')
        else:
            ff = outf.create_dataset(dset, (0, nvars), float,
                                     chunks=(128, nvars),
                                     maxshape=(None, nvars))
            ff.dims[1].label = self._header

        self._forces = DatasetAppender(ff)
        self._write = self._write_hdf5

    def _write_hdf5(self, t, forces):
        self._forces(np.concatenate(([t], forces.ravel())))

    def _init_kernels(self, intg):
        backend = self.backend

        # Register our kernel template
        backend.pointwise.register('pyfr.plugins.kernels.fluidforce')

        # Precompute per-face data and upload to device
        fi = self.ff_int
        self._efaces = []

        for (etype, fidx), m0 in fi.m0.items():
            nfpts, nupts = m0.shape
            eidxs = fi.eidxs[etype, fidx]
            neles = len(eidxs)

            # Weighted normals: qwts * norms → (nfpts, ndims, neles)
            qwts, norms = fi.qwts[etype, fidx], fi.norms[etype, fidx]
            wnorms = (qwts[None, :, None]*norms).transpose(1, 2, 0)

            # Moment arm positions at face points
            if self._mcomp:
                rfpts = fi.rfpts[etype, fidx].transpose(1, 2, 0)
                rfpts_mat = backend.const_matrix(rfpts, tags={'align'})
            else:
                rfpts_mat = None

            self._efaces.append({
                'idx': self._etype_map[etype],
                'eidxs': eidxs,
                'm0': m0, 'nupts': nupts, 'nfpts': nfpts,
                'wnorms': backend.const_matrix(wnorms, tags={'align'}),
                'rfpts': rfpts_mat,
                'pf': backend.matrix((self._nout, neles), tags={'align'}),
            })

    @memoize
    def _get_kerns(self, uidx):
        kerns = []

        for ef in self._efaces:
            eidxs = ef['eidxs']
            nupts, nfpts = ef['nupts'], ef['nfpts']

            tplargs = {
                'ndims': self.ndims, 'nvars': self.nvars,
                'nupts': nupts, 'nfpts': nfpts, 'nout': self._nout,
                'viscous': self._viscous, 'visc_corr': self._viscorr,
                'mcomp': self._mcomp, 'c': self._constants,
                'm0': ef['m0'],
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
                tplargs=tplargs, dims=[len(eidxs)],
                u=u, gradu=gradu, wnorms=ef['wnorms'],
                rfpts=ef['rfpts'], pf=ef['pf']
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

            self._write(intg.tcurr, fm)

            # Publish force components
            pn = ['px', 'py', 'pz'][:self.ndims]
            pvals = dict(zip(pn, fm[0]))
            if self._viscous:
                vn = ['vx', 'vy', 'vz'][:self.ndims]
                pvals.update(zip(vn, fm[-1]))

            self._publish(intg, **pvals)
