import h5py
import numpy as np

from pyfr.cache import memoize
from pyfr.mpiutil import get_comm_rank_root
from pyfr.plugins.common import DatasetAppender, init_csv, open_hdf5_a
from pyfr.plugins.mixins import BackendMixin
from pyfr.plugins.soln.base import BaseSolnPlugin
from pyfr.points import PointSampler
from pyfr.util import first


class SamplerPlugin(BackendMixin, BaseSolnPlugin):
    name = 'sampler'
    systems = ['*']
    dimensions = [2, 3]

    def __init__(self, intg, cfgsect, suffix):
        super().__init__(intg, cfgsect, suffix)

        # MPI info
        comm, rank, root = get_comm_rank_root()

        # Output format and gradient sampling
        self.fmt = self.cfg.get(cfgsect, 'format', 'primitive')
        self._sample_grads = self.cfg.getbool(cfgsect, 'sample-gradients',
                                              False)

        # Get the sample points
        spts = self.cfg.get(self.cfgsect, 'samp-pts')
        if ',' in spts:
            spts = self.cfg.getliteral(self.cfgsect, 'samp-pts')

        # Number of output variables per sample
        self.nsvars = (1 + self.ndims*self._sample_grads)*self.nvars

        # Construct and configure the point sampler (for location + MPI)
        self.psampler = PointSampler(intg.system.mesh, spts)
        self.psampler.configure_with_intg_nvars(intg, self.nsvars)

        # Initialise backend and kernel infrastructure
        self._init_backend(intg)
        self._init_kernels(intg)

        # Have the root rank open the output file
        if rank == root:
            match self.cfg.get(cfgsect, 'file-format', 'csv'):
                case 'csv':
                    self._init_csv(intg)
                case 'hdf5':
                    self._init_hdf5(intg)
                case _:
                    raise ValueError('Invalid file format')

    def _init_kernels(self, intg):
        backend = self.backend

        # Register our kernel template
        backend.pointwise.register('pyfr.plugins.kernels.sample')

        # Common template arguments
        self._tplargs_common = {
            'ndims': self.ndims, 'nvars': self.nvars, 'nsvars': self.nsvars,
            'has_grads': self._sample_grads,
            'primitive': self.fmt == 'primitive',
            'c': self.cfg.items_as('constants', float),
            'eos_mod': self._eos_mod
        }

        # Build per-element-type data structures
        self._edata = []
        for et, (eidxs, wts, smap) in self.psampler.etype_pinfo().items():
            npts = len(eidxs)
            nupts = intg.system.ele_map[intg.system.ele_types[et]].nupts

            # Gradient view (bank-independent, so created once here)
            if self._sample_grads:
                gradu = self._make_view(self._grad_banks[et], eidxs,
                                        (self.ndims*nupts, self.nvars))
            else:
                gradu = None

            self._edata.append({
                'idx': et,
                'eidxs': eidxs,
                'nupts': nupts,
                'wts': backend.const_matrix(wts.T, tags={'align'}),
                'out': backend.matrix((self.nsvars, npts), tags={'align'}),
                'gradu': gradu,
                'map': smap,
            })

    @memoize
    def _get_kerns(self, uidx):
        kerns = []

        for ed in self._edata:
            eidxs = ed['eidxs']
            nupts = ed['nupts']

            tplargs = {**self._tplargs_common, 'nupts': nupts}

            # Solution view into scal_upts[uidx]
            u = self._make_view(self._ele_banks[ed['idx']][uidx], eidxs,
                                (nupts, self.nvars))

            kerns.append(self.backend.pointwise.sample(
                tplargs=tplargs, dims=[len(eidxs)],
                u=u, gradu=ed['gradu'], wts=ed['wts'], out=ed['out']
            ))

        return kerns

    def _init_csv(self, intg):
        self.csv = init_csv(self.cfg, self.cfgsect, self._header(intg),
                            nflush=len(self.psampler.pts))
        self._write = self._write_csv

    def _write_csv(self, t, samps):
        for ploc, samp in zip(self.psampler.pts, samps):
            self.csv(t, *ploc, *samp)

    def _init_hdf5(self, intg):
        outf = open_hdf5_a(self.cfg.get(self.cfgsect, 'file'))

        pts = self.psampler.pts
        npts, nsvars = len(pts), self.nsvars
        chunk = 128

        dset = self.cfg.get(self.cfgsect, 'file-dataset')
        if dset in outf:
            d = outf[dset]
            t = d.dims[0][0]
            p = d.dims[1][0]

            # Ensure the point sets are compatible
            if p.shape != pts.shape or not np.allclose(p[:], pts):
                raise ValueError('Inconsistent sample points')
        else:
            d = outf.create_dataset(dset, (0, npts, nsvars), float,
                                    chunks=(chunk, min(4, npts), nsvars),
                                    maxshape=(None, npts, nsvars))
            t = outf.create_dataset(f'{dset}_t', (0,), float, chunks=(chunk,),
                                    maxshape=(None,))
            p = outf.create_dataset(f'{dset}_p', data=pts)

            t.make_scale('t')
            p.make_scale('pts')
            d.dims[0].attach_scale(t)
            d.dims[1].attach_scale(p)

        self._t = DatasetAppender(t)
        self._samps = DatasetAppender(d)
        self._write = self._write_hdf5

    def _write_hdf5(self, t, samps):
        self._t(t)
        self._samps(samps)

    def _header(self, intg):
        eles = first(intg.system.ele_map.values())
        dims = 'xyz'[:self.ndims]

        vmap = eles.privars if self.fmt == 'primitive' else eles.convars

        colnames = ['t', *dims, *vmap]
        if self._sample_grads:
            colnames.extend(f'grad_{v}_{d}' for v in vmap for d in dims)

        return ','.join(colnames)

    def __call__(self, intg):
        # Compute gradients on device if needed
        if self._sample_grads:
            intg.compute_grads()

        # Run sampling kernels on device
        self.backend.run_kernels(self._get_kerns(intg.idxcurr))

        # Collect results from device
        samples = np.empty((self.psampler.pcount, self.nsvars))
        for ed in self._edata:
            samples[ed['map']] = ed['out'].get().T

        # Gather to root rank and output
        samps = self.psampler.gather(samples)
        if samps is not None:
            self._write(intg.tcurr, samps)
