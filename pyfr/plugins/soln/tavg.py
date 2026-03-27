import re

import numpy as np

from pyfr.cache import memoize
from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.nputil import npeval
from pyfr.plugins.mixins import BackendMixin, PostactionMixin, RegionMixin
from pyfr.plugins.soln.base import BaseSolnPlugin
from pyfr.plugins.fieldeval import compile_expr
from pyfr.writers.native import NativeWriter
from pyfr.util import first


class TavgMixin:
    @staticmethod
    def _fwd_diff(f, x, axis=0):
        fx = f(x)
        dfx = np.empty((len(x), *fx.shape), dtype=fx.dtype)

        for xi, dfxi in zip(x, dfx):
            # Calculate step size for finite difference
            hi = np.finfo(x.dtype).eps**0.5*np.abs(xi)
            hi[np.where(hi == 0)] = np.finfo(x.dtype).eps**0.5

            # Apply the differencing
            xi += hi
            dfxi[:] = (f(x) - fx) / np.expand_dims(hi, axis)
            xi -= hi

        return fx, dfx


class TavgPlugin(PostactionMixin, RegionMixin, BackendMixin, TavgMixin,
                 BaseSolnPlugin):
    name = 'tavg'
    systems = ['*']
    dimensions = [2, 3]

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        comm, rank, root = get_comm_rank_root()

        # Initialise backend infrastructure
        self._init_backend(intg)

        # Primitive variables
        self.privars = first(intg.system.ele_map.values()).privars

        # Averaging mode
        self.mode = self.cfg.get(cfgsect, 'mode', 'windowed')
        if self.mode not in {'continuous', 'windowed'}:
            raise ValueError('Invalid averaging mode')

        # Std deviation mode
        self.std_mode = self.cfg.get(cfgsect, 'std-mode', 'summary')
        if self.std_mode not in {'summary', 'all', 'none'}:
            raise ValueError('Invalid standard deviation mode')

        # Expressions pre-processing
        nfields = self._prepare_exprs()

        # Output data type
        fpdtype = self.cfg.get(cfgsect, 'precision', 'single')
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

        # Determine what gradients, if any, are required
        self._has_grads = bool(set().union(*(
            re.findall(r'\bgrad_(.+?)_[xyz]\b', ex) for ex in self.aexprs
        )))

        # Time averaging parameters
        self.dtout = self.cfg.getfloat(cfgsect, 'dt-out')
        self._accum_nsteps = self.nsteps
        self.nsteps = None

        # Register our output times with the integrator
        intg.call_plugin_dt(intg.tcurr, self.dtout)

        # Mark ourselves as not currently averaging
        self._started = False

        # Determine total region points and per-type view usage
        self.tpts, self._use_views = 0, {}
        for etype, eidxs in self._ele_region_data.items():
            self.tpts += len(eidxs)*emap[etype].nupts
            self._use_views[etype] = len(eidxs) < emap[etype].neles

        # Reduce
        self.tpts = comm.reduce(self.tpts, op=mpi.SUM, root=root)

        # Check if we are restarting
        if intg.isrestart:
            self.tout_last = intg.tcurr
        else:
            self.tout_last = None

        # Initialize JIT kernel infrastructure
        self._init_kernels(intg)

    def _prepare_exprs(self):
        cfg, cfgsect = self.cfg, self.cfgsect
        c = self.cfg.items_as('constants', float)
        self.anames, self.aexprs = [], []
        self.fnames, self.fexprs = [], []

        # Iterate over accumulation expressions first
        for k in cfg.items(cfgsect, prefix='avg-'):
            self.anames.append(k.removeprefix('avg-'))
            self.aexprs.append(cfg.getexpr(cfgsect, k, subs=c))

        # Followed by any functional expressions
        for k in cfg.items(cfgsect, prefix='fun-avg-'):
            self.fnames.append(k.removeprefix('fun-avg-'))
            self.fexprs.append(cfg.getexpr(cfgsect, k, subs=c))

        # Build grouped field structure for the nested dtype
        self.field_groups = fg = {'avg': self.anames}
        if self.fnames:
            fg['fun-avg'] = self.fnames
        if self.std_mode == 'all':
            fg['avg-std'] = self.anames
            if self.fnames:
                fg['fun-avg-std'] = self.fnames

        return sum(len(v) for v in fg.values())

    def _init_kernels(self, intg):
        backend, emap = self.backend, intg.system.ele_map

        # Register the tavg kernel module
        backend.pointwise.register('pyfr.plugins.kernels.tavg')

        # Compile expressions to C-style
        cexprs = [compile_expr(e, self.privars, self.ndims)
                  for e in self.aexprs]

        # Determine accumulator dtype: use double if backend supports it
        self._use_kahan = use_kahan = not backend.has_double
        self._acc_dtype = backend.fpdtype if use_kahan else np.float64
        has_var = self.std_mode != 'none'

        # Common template arguments for the kernel
        tplargs_common = {
            'ndims': self.ndims, 'nvars': self.nvars,
            'nexprs': len(self.aexprs), 'exprs': cexprs,
            'c': self.cfg.items_as('constants', float),
            'has_grads': self._has_grads, 'has_var': has_var,
            'use_kahan': use_kahan, 'eos_mod': self._eos_mod,
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
            vacc = amat(self._acc_dtype) if has_var else None
            acc_comp = amat(backend.fpdtype) if use_kahan else None

            tplargs = {**tplargs_common, 'use_views': use_views}
            self._tavg_data[etype] = dict(
                idx=self._etype_map[etype],
                acc=acc, vacc=vacc, prev=prev, acc_comp=acc_comp,
                nupts=nupts, neles=neles, tplargs=tplargs
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
                tplargs=d['tplargs'],
                dims=[d['nupts'], d['neles']],
                u=u, gradu=gradu,
                acc=d['acc'], acc_comp=d['acc_comp'],
                vacc=d['vacc'], prev=d['prev']
            ))

        return kerns

    def _init_accumex(self, intg):
        self.tstart_acc = self.prevt = intg.tcurr

        # Don't change tout_last if we are restarting past tstart
        if self._started or self.tout_last is None:
            self.tout_last = intg.tcurr

        # Initialize host arrays for output processing
        nexprs = len(self.aexprs)
        self.accex = [np.zeros((nexprs, d['nupts'], d['neles']))
                      for d in self._tavg_data.values()]
        self.vaccex = [np.zeros_like(a) for a in self.accex]

    def _eval_fun_avg(self, avars):
        subs = dict(zip(self.anames, avars))

        # Evaluate the function and return
        return np.array([npeval(v, subs) for v in self.fexprs])

    def _eval_fun_avg_var(self, dev, accex):
        exprs, dexprs = [], []

        # Iterate over the element types
        for av in accex:
            # Apply forward differencing
            f, df = self._fwd_diff(self._eval_fun_avg, av)
            exprs.append(f)
            dexprs.append(df)

        # Multiply by variance and take RMS value
        fvar = [np.linalg.norm(df*sd[:, None], axis=0)
                for df, sd in zip(dexprs, dev)]
        return exprs, fvar

    def _prepare_meta(self, intg, std_max, std_sum):
        comm, rank, root = get_comm_rank_root()

        stats = Inifile()
        stats.set('data', 'prefix', 'tavg')
        stats.set('tavg', 'cfg-section', self.cfgsect)
        stats.set('tavg', 'range', f'[({self.tstart_acc}, {intg.tcurr})]')

        intg.collect_stats(stats)

        # Reduce our standard deviations across ranks
        if std_max is not None:
            if rank != root:
                comm.Reduce(std_max, None, op=mpi.MAX, root=root)
                comm.Reduce(std_sum, None, op=mpi.SUM, root=root)
                return None

            comm.Reduce(mpi.IN_PLACE, std_max, op=mpi.MAX, root=root)
            comm.Reduce(mpi.IN_PLACE, std_sum, op=mpi.SUM, root=root)

            names = [*self.anames, *(f'fun-{n}' for n in self.fnames)]
            for n, m, s in zip(names, std_max, std_sum):
                stats.set('tavg', f'max-std-{n}', m)
                stats.set('tavg', f'avg-std-{n}', s / self.tpts)
        elif rank != root:
            return None

        return {**intg.cfgmeta, 'stats': stats.tostr(),
                'mesh-uuid': intg.mesh_uuid}

    def _prepare_data(self, intg):
        nacc, nfun = len(self.anames), len(self.fnames)

        wts = 2*(intg.tcurr - self.tstart_acc)

        # Normalise the accumulated averages
        avg = [a / wts for a in self.accex]

        # Calculate standard deviations if tracking variance
        if self.std_mode == 'none':
            dev = std_max = std_sum = None
        else:
            std_max, std_sum = np.zeros((2, nacc + nfun))

            dev = [np.sqrt(np.abs(v / wts)) for v in self.vaccex]
            for dx in dev:
                np.maximum(np.amax(dx, axis=(1, 2)), std_max[:nacc],
                           out=std_max[:nacc])
                std_sum[:nacc] += dx.sum(axis=(1, 2))

        # Handle any functional expressions
        if not self.fexprs:
            funavg = fundev = None
        elif dev is not None:
            funavg, fundev = self._eval_fun_avg_var(dev, avg)
            for fx in fundev:
                np.maximum(np.amax(fx, axis=(1, 2)),
                           std_max[nacc:], out=std_max[nacc:])
                std_sum[nacc:] += fx.sum(axis=(1, 2))
        else:
            funavg = [self._eval_fun_avg(a) for a in avg]
            fundev = None

        # Build grouped data per element type
        data = {}
        for i, (idx, etype, rgn) in enumerate(self._ele_regions):
            data[etype] = d = {'avg': avg[i].transpose(2, 0, 1)}
            if funavg is not None:
                d['fun-avg'] = funavg[i].transpose(2, 0, 1)
            if self.std_mode == 'all':
                d['avg-std'] = dev[i].transpose(2, 0, 1)
                if fundev is not None:
                    d['fun-avg-std'] = fundev[i].transpose(2, 0, 1)

        return data, self._prepare_meta(intg, std_max, std_sum)

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

            # Compute weights for trapezoidal rule + Welford variance
            dt = intg.tcurr - self.prevt
            wacc = intg.tcurr - self.tstart_acc
            wvar = 2*(wacc - dt)*wacc if self.tstart_acc != self.prevt else 0.0

            # Bind weights and run accumulation kernels
            kerns = self._get_accum_kerns(intg.idxcurr)
            for kern in kerns:
                kern.bind(wdt=dt, wacc=wacc, wvar=wvar)
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
            if d['vacc'] is not None:
                kerns.append(self.backend.kernel('zero', d['vacc']))
            if d['acc_comp'] is not None:
                kerns.append(self.backend.kernel('zero', d['acc_comp']))

        return kerns

    def _zero_accumulators(self):
        self.backend.run_kernels(self._get_zero_kerns())

    def _fetch_accumulators(self):
        for i, d in enumerate(self._tavg_data.values()):
            for src, dst in [(d['acc'], self.accex), (d['vacc'], self.vaccex)]:
                if src is not None:
                    dst[i][:] = src.get().transpose(1, 0, 2)

    def finalise(self, intg):
        super().finalise(intg)

        self._writer.flush()
