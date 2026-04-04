import re

import h5py
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured as s2u

from pyfr.cache import memoize
from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.nputil import npeval
from pyfr.plugins.base import (BackendMixin, BaseCLIPlugin, BaseSolnPlugin,
                               PostactionMixin, RegionMixin, cli_external)
from pyfr.plugins.fieldeval import compile_expr
from pyfr.progress import NullProgressBar
from pyfr.writers.native import NativeWriter
from pyfr.util import first, merge_intervals


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


class TavgCLIPlugin(TavgMixin, BaseCLIPlugin):
    name = 'tavg'

    @classmethod
    def add_cli(cls, parser):
        sp = parser.add_subparsers()

        # Merge command
        ap_merge = sp.add_parser('merge', help='tavg merge --help')
        ap_merge.set_defaults(process=cls.merge_cli)
        ap_merge.add_argument('solns', nargs='*', help='averages to merge')
        ap_merge.add_argument('output', help='output file name')

    @cli_external
    def merge_cli(self, args):
        # Open all the solution files
        with args.progress.start('Preprocess files'):
            self._preprocess_files(args.solns)

            # Initialise things needed for the merge
            self._init_tavg_merge()

        with h5py.File(args.output, 'w', libver='latest') as outf:
            with args.progress.start('Prepare output file'):
                self._prepare_output_file(outf)

            # Merge the averages
            with args.progress.start_with_bar('Merge data') as pbar:
                self._merge_data(outf, pbar)

            # Merge the metadata
            with args.progress.start('Merge metadata'):
                self._merge_stats(outf)

    def _eval_fun_avg(self, avars):
        subs = dict(zip(self.anames, avars))
        return np.stack([npeval(v, subs) for v in self.fexprs],
                        axis=1)

    def _eval_fun_avg_var(self, acc, std):
        acc, std = acc.swapaxes(0, 1), std.swapaxes(0, 1)
        favg, dfavg = self._fwd_diff(self._eval_fun_avg, acc, axis=1)
        return favg, np.linalg.norm(dfavg*std[:, :, None], axis=0)

    def _init_tavg_merge(self):
        f0, cfg0, stats0, _ = self.files[0]
        cfgsect = stats0.get('tavg', 'cfg-section')

        self.cfg, self.stats, self.cfgsect = cfg0, stats0, cfgsect
        self.region = cfg0.get(cfgsect, 'region')
        self.uuid = f0['mesh-uuid'][()].decode()

        # Extract record dtypes, dataset shapes, and point count
        dshapes, self._dtypes, self.tpts = {}, {}, 0
        for k, v in f0['tavg'].items():
            if re.match(r'p\d+-[a-z]+$', k):
                dshapes[f'tavg/{k}'] = v.shape
                self._dtypes[k] = v.dtype
                self.tpts += v.shape[0]*v.dtype[0][0].shape[0]

        # Use first element type's dtype for group/field discovery
        dtype0 = first(self._dtypes.values())
        self.has_fun = 'fun-avg' in dtype0.names

        # Expression names and compiled expressions
        c = cfg0.items_as('constants', float)
        if self.has_fun:
            self.fnames = sorted(dtype0['fun-avg'].names)
            self.fexprs = [cfg0.getexpr(cfgsect, f'fun-avg-{n}', subs=c)
                           for n in self.fnames]
        else:
            self.fnames, self.fexprs = [], []

        # Compute common avg fields across all files
        fset = set(dtype0['avg'].names)

        for f, cfg, stats, _ in self.files[1:]:
            cs = stats.get('tavg', 'cfg-section')
            if self.uuid != f['mesh-uuid'][()].decode():
                raise RuntimeError('Files from different meshes')
            if self.region != cfg.get(cs, 'region'):
                raise RuntimeError('Files from different regions')
            for k in cfg.items(cs, prefix='avg-'):
                if cfg.get(cs, k) != cfg0.get(cfgsect, k):
                    raise RuntimeError('Different average field definitions')
            dt = first(v.dtype for k, v in f['tavg'].items()
                       if re.match(r'p\d+-[a-z]+$', k))
            fset &= set(dt['avg'].names)

        self._afields = self.anames = sorted(fset)

        # Standard deviation tracking
        if self.std_all:
            self.std_max = np.zeros(len(self._afields))
            self.std_sum = np.zeros(len(self._afields))

            if self.has_fun:
                self.fstd_max = np.zeros(len(self.fnames))
                self.fstd_sum = np.zeros(len(self.fnames))

        # Break each dataset into ~2 GiB chunks
        chunk_sz = -(2*1024**3 // -dtype0.itemsize)
        self.chunks = [(k, slice(i, i + chunk_sz))
                       for k, v in dshapes.items()
                       for i in range(0, v[0], chunk_sz)]

    def _prepare_output_file(self, outf):
        basef = self.files[0][0]

        # Copy over top level records except stats and tavg
        for k, v in basef.items():
            if k not in ('stats', 'tavg'):
                basef.copy(v, outf, k)

        # Handle the tavg group
        for k, v in basef['tavg'].items():
            if re.match(r'p\d+-\w+-', k):
                basef.copy(v, outf, f'tavg/{k}')
            else:
                dt = self._odtype(v.dtype)
                w = outf.create_dataset(f'tavg/{k}', v.shape, dt)
                for ak, av in v.attrs.items():
                    w.attrs[ak] = av

    def _odtype(self, idtype):
        groups = [('avg', self._afields)]

        if self.has_fun:
            groups.append(('fun-avg', self.fnames))

        if self.std_all:
            groups.append(('avg-std', self._afields))
            if self.has_fun:
                groups.append(('fun-avg-std', self.fnames))

        return np.dtype([(g, [(fn, idtype[g][fn]) for fn in fns])
                         for g, fns in groups])

    def _unpack(self, data, group):
        dg = data[group][self._afields]
        return s2u(dg).reshape(len(data), len(dg.dtype), -1)

    def _merge_data(self, outf, pbar=NullProgressBar()):
        file0 = self.files[0][0]

        for k, s in pbar.start_with_iter(self.chunks):
            idtype = self._dtypes[k.split('/')[-1]]

            # Initialise accumulators
            shape = self._unpack(file0[k][s], 'avg').shape
            acc = np.zeros(shape, dtype=float)
            var = np.zeros(shape, dtype=float) if self.std_all else None

            # Merge the base averages and variances
            t = 0
            for file, *_, w in self.files:
                d = self._unpack(file[k][s], 'avg')
                if var is not None:
                    ds = self._unpack(file[k][s], 'avg-std')
                    if t > 0:
                        delta = d - acc / t
                        tw = t + w
                        var = (t*var + w*ds**2)/tw + t*w/tw**2*delta**2
                    else:
                        var = ds**2

                acc += w*d
                t += w

            # Build output record
            out = np.empty(acc.shape[0], dtype=self._odtype(idtype))

            # Pack merged averages
            for fn, col in zip(self._afields, acc.swapaxes(0, 1)):
                out['avg'][fn] = col

            # Finalise standard deviations and function expressions
            stds = []
            if self.std_all:
                std = np.sqrt(np.abs(var))
                stds.append((self.std_max, self.std_sum,
                             self._afields, 'avg-std', std))

            if self.fexprs:
                if self.std_all:
                    favg, fstd = self._eval_fun_avg_var(acc, std)
                    stds.append((self.fstd_max, self.fstd_sum,
                                 self.fnames, 'fun-avg-std', fstd))
                else:
                    favg = self._eval_fun_avg(acc.swapaxes(0, 1))

                for fn, col in zip(self.fnames, favg.swapaxes(0, 1)):
                    out['fun-avg'][fn] = col

            for smax, ssum, names, group, sd in stds:
                for i, col in enumerate(sd.swapaxes(0, 1)):
                    smax[i] = max(smax[i], col.max())
                    ssum[i] += col.sum()
                    out[group][names[i]] = col

            outf[k][s] = out

    def _merge_stats(self, outf):
        nstats = Inifile()

        # Create the data block
        nstats.set('data', 'prefix', 'tavg')

        # Create the tavg block
        cfgsect = self.stats.get('tavg', 'cfg-section')
        nstats.set('tavg', 'cfg-section', cfgsect)
        nstats.set('tavg', 'range', self.merged_range)
        nstats.set('tavg', 'merged-from', self.merged_from)

        # If all files have full std stats then these can be written
        if self.std_all:
            for n, s, m in zip(self._afields, self.std_sum, self.std_max):
                nstats.set('tavg', f'avg-std-{n}', s / self.tpts)
                nstats.set('tavg', f'max-std-{n}', m)

            if self.has_fun:
                for n, s, m in zip(self.fnames, self.fstd_sum, self.fstd_max):
                    nstats.set('tavg', f'avg-std-fun-{n}', s / self.tpts)
                    nstats.set('tavg', f'max-std-fun-{n}', m)

        # Write out the new stats record
        outf['stats'] = np.array(nstats.tostr().encode(), dtype='S')

    def _preprocess_files(self, filenames):
        files, twindows, std_all = [], [], True

        for filename in filenames:
            f = h5py.File(filename, 'r')
            cfg = Inifile(f['config'][()].decode())
            stats = Inifile(f['stats'][()].decode())
            cfgsect = stats.get('tavg', 'cfg-section')

            twind = stats.getliteral('tavg', 'range')
            dt = sum(te - ts for ts, te in twind)

            files.append((f, cfg, stats, dt))
            twindows.extend(twind)
            std_all &= cfg.get(cfgsect, 'std-mode') == 'all'

        self.std_all, self.merged_from = std_all, twindows

        try:
            self.merged_range = merge_intervals(twindows)
        except ValueError:
            raise RuntimeError('Overlapping average time ranges in files')

        avg_time = sum(te - ts for ts, te in self.merged_range)
        self.files = [(*fcs, dt / avg_time) for *fcs, dt in files]
