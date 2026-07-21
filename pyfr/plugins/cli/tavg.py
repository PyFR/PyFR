import re

import h5py
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured as s2u

from pyfr.inifile import Inifile
from pyfr.plugins.base import BaseCLIPlugin
from pyfr.plugins.common import cli_external, get_elementscls
from pyfr.progress import NullProgressBar
from pyfr.shapes import BaseShape
from pyfr.stats import eval_algebraic, tavg_exprs
from pyfr.stats.provider import name_tokens
from pyfr.util import first, merge_intervals, subclass_where


class TavgCLIPlugin(BaseCLIPlugin):
    name = 'tavg'

    @classmethod
    def add_cli(cls, parser):
        sp = parser.add_subparsers()

        # Merge command
        ap_merge = sp.add_parser('merge', help='tavg merge --help')
        ap_merge.set_defaults(process=cls.merge_cli)
        ap_merge.add_argument('solns', nargs='+', help='averages to merge')
        ap_merge.add_argument('output', help='output file name')
        ap_merge.add_argument('-r', '--report', action='store_true',
                              help='report window-to-window convergence')

    @cli_external
    def merge_cli(self, args):
        self.report = args.report

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

        if self.report:
            self._report_convergence()

    def _eval_derived(self, avars):
        fields = dict(zip(self.anames, avars))
        out = eval_algebraic(self._alg, fields, self._hidden)

        return np.stack(list(out.values()), axis=1)

    def _init_tavg_merge(self):
        f0, cfg0, stats0, _ = self.files[0]
        cfgsect = stats0.get('tavg', 'cfg-section')

        self.cfg, self.stats, self.cfgsect = cfg0, stats0, cfgsect
        self.region = cfg0.get(cfgsect, 'region')
        self.uuid = f0['mesh-uuid'][()].decode()

        # Extract record dtypes and dataset shapes
        dshapes, self._dtypes = {}, {}
        etypes = set()
        for k, v in f0['tavg'].items():
            if (m := re.fullmatch(r'p\d+-([a-z]+)', k)):
                dshapes[f'tavg/{k}'] = v.shape
                self._dtypes[k] = v.dtype
                etypes.add(m[1])

        # Deduce the dimensionality from the element types
        ndims = subclass_where(BaseShape, name=first(etypes)).ndims

        # Use first element type's dtype for group/field discovery
        dtype0 = first(self._dtypes.values())

        # Derive the lowered expressions from a file's stored config
        def exprs_of(cfg, cs):
            return tavg_exprs(cfg, cs, ndims, get_elementscls(cfg))

        te0 = exprs_of(cfg0, cfgsect)
        exprs0 = te0.avgs

        # Compute common avg fields across all files
        fset = set(dtype0['avg'].names)

        for f, cfg, stats, _ in self.files[1:]:
            cs = stats.get('tavg', 'cfg-section')
            if self.uuid != f['mesh-uuid'][()].decode():
                raise RuntimeError('Files from different meshes')
            if self.region != cfg.get(cs, 'region'):
                raise RuntimeError('Files from different regions')
            dt = first(v.dtype for k, v in f['tavg'].items()
                       if re.fullmatch(r'p\d+-[a-z]+', k))
            fset &= set(dt['avg'].names)

            # Common fields must have identical lowered definitions
            exprs = exprs_of(cfg, cs).avgs
            for k in fset:
                if exprs.get(k) != exprs0.get(k):
                    raise RuntimeError('Different average field definitions')

        self.anames = sorted(fset)

        # Window-to-window convergence accumulators
        if self.report:
            if len(self.files) < 2:
                raise RuntimeError('Convergence reporting requires at '
                                   'least two windows')

            # Algebraic derived quantities evaluable from the common fields
            known, hid = set(te0.avgs), te0.hidden

            def evaluable(e):
                for tok in name_tokens(e):
                    if tok in known and tok not in fset:
                        return False
                    if tok in hid and not evaluable(hid[tok]):
                        return False
                return True

            self._alg = {n: e for n, e in te0.derived.items()
                         if n in te0.alg and evaluable(e)}
            self._hidden = {n: e for n, e in hid.items()
                            if n in te0.alg and evaluable(e)}
            self._rep = {g: np.zeros((2, len(ns))) for g, ns in
                         [('avg', self.anames), ('derived', self._alg)] if ns}

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
        sdt = idtype['avg'][self.anames[0]]

        return np.dtype([('avg', [(fn, sdt) for fn in self.anames])])

    def _unpack(self, data, group):
        dg = data[group][self.anames]
        return s2u(dg).reshape(len(data), len(dg.dtype), -1)

    def _merge_data(self, outf, pbar=NullProgressBar()):
        file0 = self.files[0][0]

        for k, s in pbar.start_with_iter(self.chunks):
            # Initialise the accumulator
            shape = self._unpack(file0[k][s], 'avg').shape
            acc = np.zeros(shape, dtype=float)

            # Merge the base averages
            t = 0
            for file, *_, w in self.files:
                acc += w*self._unpack(file[k][s], 'avg')
                t += w

            # Accumulate window-to-window convergence statistics
            if self.report:
                self._accumulate_report(k, s, acc / t)

            # Build output record
            out = np.empty(acc.shape[0], dtype=outf[k].dtype)

            # Pack merged averages
            for fn, col in zip(self.anames, acc.swapaxes(0, 1)):
                out['avg'][fn] = col

            outf[k][s] = out

    def _accumulate_report(self, k, s, m):
        rep = self._rep
        rep['avg'][1] += (m*m).sum(axis=(0, 2))

        if 'derived' in rep:
            mf = self._eval_derived(m.swapaxes(0, 1))
            rep['derived'][1] += (mf*mf).sum(axis=(0, 2))

        # Weighted squared deviation of each window from the merged mean
        for file, *_, w in self.files:
            dev = self._unpack(file[k][s], 'avg') - m
            rep['avg'][0] += w*w*(dev*dev).sum(axis=(0, 2))

            if 'derived' in rep:
                df = self._eval_derived((dev + m).swapaxes(0, 1)) - mf
                rep['derived'][0] += w*w*(df*df).sum(axis=(0, 2))

    def _report_convergence(self):
        # Batch-means small sample correction factor
        n = len(self.files)
        fac = 1/(1 - sum(w*w for *_, w in self.files))

        print(f'Relative standard error over {n} windows')
        gnames = {'avg': self.anames, 'derived': self._alg}
        for g, (se2, m2) in self._rep.items():
            rse = np.sqrt(fac*se2)/np.maximum(np.sqrt(m2), 1e-300)
            for name, r in zip(gnames[g], rse):
                print(f'{g}-{name}\t{r:.4e}')

    def _merge_stats(self, outf):
        nstats = Inifile()

        # Create the data block
        nstats.set('data', 'prefix', 'tavg')

        # Create the tavg block
        cfgsect = self.stats.get('tavg', 'cfg-section')
        nstats.set('tavg', 'cfg-section', cfgsect)
        nstats.set('tavg', 'range', self.merged_range)
        nstats.set('tavg', 'merged-from', self.merged_from)

        # Preserve any default postproc plugin request
        if self.stats.hasopt('data', 'postproc'):
            nstats.set('data', 'postproc', self.stats.get('data', 'postproc'))

        # Write out the new stats record
        outf['stats'] = np.array(nstats.tostr().encode(), dtype='S')

    def _preprocess_files(self, filenames):
        files, twindows = [], []

        for filename in filenames:
            f = h5py.File(filename, 'r')
            cfg = Inifile(f['config'][()].decode())
            stats = Inifile(f['stats'][()].decode())

            if stats.get('data', 'prefix') != 'tavg':
                raise RuntimeError(f'{filename} is not a time-average file')

            twind = stats.getliteral('tavg', 'range')
            dt = sum(te - ts for ts, te in twind)

            files.append((f, cfg, stats, dt))
            twindows.extend(twind)

        self.merged_from = twindows

        try:
            self.merged_range = merge_intervals(twindows)
        except ValueError:
            raise RuntimeError('Overlapping average time ranges in files')

        avg_time = sum(te - ts for ts, te in self.merged_range)
        self.files = [(*fcs, dt / avg_time) for *fcs, dt in files]
