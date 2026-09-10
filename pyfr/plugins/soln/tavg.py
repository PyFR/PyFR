import re

import numpy as np

from pyfr.cache import memoize
from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root
from pyfr.stats import tavg_exprs
from pyfr.plugins.fieldeval import compile_expr
from pyfr.plugins.mixins import BackendMixin, PostactionMixin, RegionMixin
from pyfr.plugins.soln.base import BaseSolnPlugin
from pyfr.util import first
from pyfr.writers.native import NativeWriter


class TavgPlugin(PostactionMixin, RegionMixin, BackendMixin, BaseSolnPlugin):
    name = 'tavg'
    systems = '.*'
    dimensions = '2|3'

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        # Initialise backend infrastructure
        self._init_backend(intg)

        # Primitive variables and the element class
        eles = first(intg.system.ele_map.values())
        self.privars = eles.privars
        self.elementscls = type(eles)

        # Averaging mode
        self.mode = self.cfg.get(cfgsect, 'mode', 'windowed')
        if self.mode not in {'continuous', 'windowed'}:
            raise ValueError('Invalid averaging mode')

        # Expressions pre-processing
        nfields = self._prepare_exprs()

        # Output data type
        prec = 'double' if self.stats_spec else 'single'
        fpdtype = self.cfg.get(cfgsect, 'precision', prec)
        fpdtype_map = {'single': np.float32, 'double': np.float64}
        self.fpdtype = fpdtype_map.get(fpdtype)
        if not self.fpdtype:
            raise ValueError('Invalid floating point data type')

        # Base output directory and file name
        basedir = self.cfg.getpath(cfgsect, 'basedir', '.', abs=True)
        basename = self.cfg.get(cfgsect, 'basename')

        # Get the element map and region data
        emap, erdata = intg.system.ele_map, self._ele_region_data

        # Figure out the shape of each element type in our region
        ershapes = {etype: (nfields, emap[etype].nupts) for etype in erdata}

        # Construct the file writer
        self._writer = NativeWriter.from_integrator(
            intg, basedir, basename, 'tavg', fpdtype=self.fpdtype
        )
        self._writer.set_shapes_eidxs(ershapes, erdata, self.field_groups)

        # Asynchronous output options
        self._async_timeout = self.cfg.getfloat(cfgsect, 'async-timeout', 60)

        # Determine if gradients are required
        self._has_grads = any(re.search(r'\bgrad_.+?_[xyz]\b', ex)
                              for ex in self.aexprs)

        if self._has_grads and not self.elementscls.has_grad_soln:
            raise ValueError('Gradients are not defined for this system')

        # Time averaging parameters
        self.dtout = self.cfg.getfloat(cfgsect, 'dt-out')
        self._accum_nsteps = self.nsteps
        self.nsteps = None

        # Register our output times with the integrator
        intg.call_plugin_dt(intg.tcurr, self.dtout)

        # Mark ourselves as not currently averaging
        self._started = False

        # Determine per-type view usage
        self._use_views = {}
        for etype, eidxs in self._ele_region_data.items():
            self._use_views[etype] = len(eidxs) < emap[etype].neles

        # Check if we are restarting
        if intg.isrestart:
            self.tout_last = intg.tcurr
        else:
            self.tout_last = None

        # Initialize JIT kernel infrastructure
        self._init_kernels(intg)

    def _prepare_exprs(self):
        cfg, cfgsect = self.cfg, self.cfgsect

        # Parse the expressions and expand any statistics packages
        te = tavg_exprs(cfg, cfgsect, self.ndims, self.elementscls)
        if not te.avgs:
            raise ValueError(f'No averages specified in [{cfgsect}]')

        self.anames, self.aexprs = list(te.avgs), list(te.avgs.values())
        self.stats_spec = te.spec

        # Derived quantities are evaluated at merge and export time
        self.has_derived = bool(te.derived)

        # Build grouped field structure for the nested dtype
        self.field_groups = {'avg': self.anames}

        return len(self.anames)

    def _init_kernels(self, intg):
        backend, emap = self.backend, intg.system.ele_map

        # Register the tavg kernel module
        backend.pointwise.register('pyfr.plugins.soln.kernels.tavg')

        # Compile expressions to C-style
        cexprs = [compile_expr(e, self.privars, self.ndims)
                  for e in self.aexprs]

        # Determine accumulator dtype: use double if backend supports it
        self._use_kahan = use_kahan = not backend.has_double
        self._acc_dtype = backend.fpdtype if use_kahan else np.float64

        # Common template arguments for the kernel
        tplargs_common = {
            'ndims': self.ndims, 'nvars': self.nvars,
            'nexprs': len(self.aexprs), 'exprs': cexprs,
            'c': self.cfg.items_as('constants', float),
            'has_grads': self._has_grads, 'use_kahan': use_kahan,
            'eos_mod': self._eos_mod
        }

        # Build per-element-type data structures
        self._tavg_data = {}
        nexprs = len(self.aexprs)

        for etype, eidxs in self._ele_region_data.items():
            eles = emap[etype]
            nupts = eles.nupts
            neles = len(eidxs) if self._use_views[etype] else eles.neles
            use_views = self._use_views[etype]

            shape = (nupts, nexprs, neles)
            amat = lambda d: backend.matrix(shape, dtype=d, tags={'align'})
            acc = amat(self._acc_dtype)
            prev = amat(backend.fpdtype)
            acc_comp = amat(backend.fpdtype) if use_kahan else None

            tplargs = {**tplargs_common, 'use_views': use_views}
            self._tavg_data[etype] = dict(
                idx=self._etype_map[etype], acc=acc, prev=prev,
                acc_comp=acc_comp, nupts=nupts, neles=neles, tplargs=tplargs
            )

    @memoize
    def _get_accum_kerns(self, uidx):
        backend = self.backend
        kerns = []

        for _, etype, rgn in self._ele_regions:
            d = self._tavg_data[etype]
            ebank = self._ele_banks[d['idx']]
            gbank = self._grad_banks[d['idx']] if self._grad_banks else None

            if self._use_views[etype]:
                nupts, nvars = d['nupts'], self.nvars
                mkv = lambda m, nr: self._make_view(m, rgn, (nr, nvars))

                u = mkv(ebank[uidx], nupts)
                gradu = mkv(gbank, self.ndims*nupts) if gbank else None
            else:
                u, gradu = ebank[uidx], gbank

            kerns.append(backend.pointwise.tavg(
                tplargs=d['tplargs'], dims=[d['nupts'], d['neles']], u=u,
                gradu=gradu, acc=d['acc'], acc_comp=d['acc_comp'],
                prev=d['prev']
            ))

        return kerns

    def _init_accumex(self, intg):
        self.tstart_acc = self.prevt = intg.tcurr

        # Reset the output clock to when averaging actually begins
        self.tout_last = intg.tcurr

        # Initialize host arrays for output processing
        nexprs = len(self.aexprs)
        self.accex = [np.zeros((nexprs, d['nupts'], d['neles']))
                      for d in self._tavg_data.values()]

    def _prepare_meta(self, intg):
        comm, rank, root = get_comm_rank_root()

        stats = Inifile()
        stats.set('data', 'prefix', 'tavg')
        stats.set('tavg', 'cfg-section', self.cfgsect)
        stats.set('tavg', 'range', f'[({self.tstart_acc}, {intg.tcurr})]')

        # Have exports finalise any derived statistics by default
        if self.has_derived:
            stats.set('data', 'postproc', 'stats')

        # Collect integrator statistics; this is collective over all ranks
        intg.collect_stats(stats)

        if rank != root:
            return None
        else:
            return {**intg.cfgmeta, 'stats': stats.tostr(),
                    'mesh-uuid': intg.mesh_uuid}

    def _prepare_data(self, intg):
        wts = 2*(intg.tcurr - self.tstart_acc)

        # Normalise the accumulated averages
        avg = [a / wts for a in self.accex]

        # Build grouped data per element type
        data = {}
        for i, (idx, etype, rgn) in enumerate(self._ele_regions):
            data[etype] = {'avg': avg[i].transpose(2, 0, 1)}

        return data, self._prepare_meta(intg)

    def trigger_write(self, intg):
        # Non-destructive snapshot: write without resetting
        if not self._started:
            return

        self._fetch_accumulators()
        data, metadata = self._prepare_data(intg)
        self._writer.write(data, intg.tcurr, metadata, self._async_timeout)

    def __call__(self, intg):
        self._writer.probe()

        # If necessary, run the start-up routines
        if not self._started:
            self._init_accumex(intg)
            self._started = True

        # See if we are due to write and/or accumulate this step
        dowrite = intg.tcurr - self.tout_last >= self.dtout - self.tol
        doaccum = intg.nacptsteps % self._accum_nsteps == 0

        if dowrite or doaccum:
            # Compute gradients on device if needed
            if self._has_grads:
                intg.compute_grads()

            # Compute the weight for the trapezoidal rule
            dt = intg.tcurr - self.prevt

            # Bind the weight and run accumulation kernels
            kerns = self._get_accum_kerns(intg.idxcurr)
            for kern in kerns:
                kern.bind(wdt=dt)
            self.backend.run_kernels(kerns)
            self.prevt = intg.tcurr

            if dowrite:
                # Transfer device data to host for output processing
                self._fetch_accumulators()

                # Prepare the data and metadata
                data, metadata = self._prepare_data(intg)

                # Prepare a callback to kick off any postactions
                callback = lambda fname, t=intg.tcurr: self._invoke_postaction(
                    intg, mesh=intg.system.mesh.fname, soln=fname, t=t
                )

                # Write out the file
                self._writer.write(data, intg.tcurr, metadata,
                                   self._async_timeout, callback)

                # Reset the accumulators
                if self.mode == 'windowed':
                    self._zero_accumulators()
                    self.tstart_acc = intg.tcurr

                self.tout_last = intg.tcurr

    @memoize
    def _get_zero_kerns(self):
        kerns = []
        for d in self._tavg_data.values():
            kerns.append(self.backend.kernel('zero', d['acc']))
            if d['acc_comp'] is not None:
                kerns.append(self.backend.kernel('zero', d['acc_comp']))

        return kerns

    def _zero_accumulators(self):
        self.backend.run_kernels(self._get_zero_kerns())

    def _fetch_accumulators(self):
        for i, d in enumerate(self._tavg_data.values()):
            self.accex[i][:] = d['acc'].get().transpose(1, 0, 2)

    def finalise(self, intg):
        self._writer.flush()

        super().finalise(intg)
