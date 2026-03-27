import re

import h5py
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured as s2u

from pyfr.inifile import Inifile
from pyfr.nputil import npeval
from pyfr.plugins.base import BaseCLIPlugin
from pyfr.plugins.common import cli_external
from pyfr.plugins.soln.tavg import TavgMixin
from pyfr.progress import NullProgressBar
from pyfr.util import first, merge_intervals


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
