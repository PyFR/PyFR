import os
import re

import h5py
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured as s2u

from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root, init_mpi, mpi
from pyfr.plugins.base import BaseCLIPlugin
from pyfr.plugins.common import cli_external, get_elementscls
from pyfr.progress import NullProgressBar, NullProgressSequence
from pyfr.shapes import BaseShape
from pyfr.stats import eval_algebraic, tavg_exprs
from pyfr.stats.provider import name_tokens
from pyfr.util import first, merge_intervals, pwrite_all, subclass_where


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
        # Initialise MPI
        init_mpi()

        # Get our MPI info
        comm, rank, root = get_comm_rank_root()

        self.report = args.report

        # Only the root rank reports progress
        progress = args.progress if rank == root else NullProgressSequence()

        # Open all the solution files
        with progress.start('Preprocess files'):
            self._preprocess_files(args.solns)

            # Initialise things needed for the merge
            self._init_tavg_merge()

        # Lay down the output file and obtain the dataset offsets
        with progress.start('Prepare output file'):
            doffs = self._prepare_output_file(args.output)

        # Merge the averages
        with progress.start_with_bar('Merge data') as pbar:
            self._merge_data(args.output, doffs, pbar)

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
            self._w2 = sum(w*w for *_, w in self.files)

        # Output record dtypes
        self._odtypes = {f'tavg/{k}': self._odtype(dt)
                         for k, dt in self._dtypes.items()}

        comm, rank, root = get_comm_rank_root()

        # Break our portion of each dataset into ~2 GiB chunks
        chunk_sz = -(2*1024**3 // -dtype0.itemsize)

        self.chunks = []
        for k, v in dshapes.items():
            sidx = (rank*v[0]) // comm.size
            eidx = ((rank + 1)*v[0]) // comm.size

            for i in range(sidx, eidx, chunk_sz):
                self.chunks.append((k, slice(i, min(i + chunk_sz, eidx))))

    def _prepare_output_file(self, path):
        comm, rank, root = get_comm_rank_root()

        # Have the root rank lay down the structure of the file
        if rank == root:
            basef, doffs = self.files[0][0], {}

            with h5py.File(path, 'w', libver='latest') as outf:
                # Copy over top level records except stats and tavg
                for k, v in basef.items():
                    if k not in ('stats', 'tavg'):
                        basef.copy(v, outf, k)

                # Handle the tavg group
                for k, v in basef['tavg'].items():
                    if re.match(r'p\d+-\w+-', k):
                        basef.copy(v, outf, f'tavg/{k}')
                    else:
                        dt = self._odtypes[f'tavg/{k}']
                        w = outf.create_dataset(f'tavg/{k}', v.shape, dt)
                        for ak, av in v.attrs.items():
                            w.attrs[ak] = av

                # Merge the metadata
                self._merge_stats(outf)

                # Allocate the average datasets and note their offsets
                for k in self._odtypes:
                    outf[k][(-1,)*outf[k].ndim] = 0
                    doffs[k] = outf[k].id.get_offset()

            return comm.bcast(doffs, root=root)
        else:
            return comm.bcast(None, root=root)

    def _odtype(self, idtype):
        sdt = idtype['avg'][self.anames[0]]

        return np.dtype([('avg', [(fn, sdt) for fn in self.anames])])

    def _unpack(self, data, group):
        dg = data[group][self.anames]
        return s2u(dg).reshape(len(data), len(dg.dtype), -1)

    def _merge_data(self, path, doffs, pbar=NullProgressBar()):
        comm, rank, root = get_comm_rank_root()

        # Open the output file for raw writing
        fd = os.open(path, os.O_WRONLY)

        try:
            self._merge_chunks(fd, doffs, pbar)
        finally:
            os.close(fd)

        # Ensure all ranks have finished writing
        comm.barrier()

    def _merge_chunks(self, fd, doffs, pbar):
        for k, s in pbar.start_with_iter(self.chunks):
            acc, t = None, 0

            # Merge the base averages
            for file, *_, w in self.files:
                d = self._unpack(file[k][s], 'avg')

                # Initialise the accumulator
                if acc is None:
                    acc = np.zeros(d.shape)
                    if self.report:
                        self._init_report_chunk(d)

                acc += w*d
                t += w

                # Accumulate window-to-window convergence statistics
                if self.report:
                    self._accumulate_report(d, w)

            # Reduce the statistics against the merged mean
            if self.report:
                self._reduce_report(acc / t)

            # Build output record
            out = np.empty(acc.shape[0], dtype=self._odtypes[k])

            # Pack merged averages
            for fn, col in zip(self.anames, acc.swapaxes(0, 1)):
                out['avg'][fn] = col

            pwrite_all(fd, doffs[k] + s.start*out.itemsize, out)

    def _init_report_chunk(self, d0):
        # Shift the deviation sums by the first window of the chunk
        self._ref = ref = d0.astype(float)
        self._dev = {'avg': np.zeros((2, *ref.shape))}

        if 'derived' in self._rep:
            self._fref = self._eval_derived(ref.swapaxes(0, 1))
            self._dev['derived'] = np.zeros((2, *self._fref.shape))

    def _accumulate_report(self, d, w):
        dev, ww, e = self._dev, w*w, d - self._ref

        # Weighted first and second moments of the shifted deviations
        dev['avg'][0] += ww*e
        dev['avg'][1] += ww*e*e

        if 'derived' in dev:
            g = self._eval_derived((self._ref + e).swapaxes(0, 1))
            g -= self._fref

            dev['derived'][0] += ww*g
            dev['derived'][1] += ww*g*g

    def _reduce_report(self, m):
        rep, c = self._rep, self._w2
        mu, (e1, e2) = m - self._ref, self._dev['avg']

        # Weighted squared deviation of each window from the merged mean
        rep['avg'][0] += (e2 - 2*mu*e1 + c*mu*mu).sum(axis=(0, 2))
        rep['avg'][1] += (m*m).sum(axis=(0, 2))

        if 'derived' in rep:
            mf = self._eval_derived(m.swapaxes(0, 1))
            nu, (g1, g2) = mf - self._fref, self._dev['derived']

            rep['derived'][0] += (g2 - 2*nu*g1 + c*nu*nu).sum(axis=(0, 2))
            rep['derived'][1] += (mf*mf).sum(axis=(0, 2))

    def _report_convergence(self):
        comm, rank, root = get_comm_rank_root()

        # Reduce the statistics accumulated by each rank
        for v in self._rep.values():
            comm.Allreduce(mpi.IN_PLACE, v, op=mpi.SUM)

        # Batch-means small sample correction factor
        n = len(self.files)
        fac = 1/(1 - self._w2)

        if rank == root:
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
