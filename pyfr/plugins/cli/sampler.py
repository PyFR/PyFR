import csv
from functools import partial
import io
from pathlib import Path
import re
import sys

import h5py
import numpy as np

from pyfr.exprs import expr_vars, npeval
from pyfr.fields import con_block_to_pri
from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root, init_mpi
from pyfr.plugins.base import BaseCLIPlugin
from pyfr.plugins.common import cli_external, get_elementscls
from pyfr.plugins.postproc import get_source
from pyfr.points import PointLocator, PointSampler
from pyfr.readers.native import NativeReader
from pyfr.series import (acf, csd, mean_ci, mser, psd, structfun, tone,
                         tw_mean_var, xcf)
from pyfr.stats import eval_algebraic, expand_exprs, soln_exprs


class _Series:
    def __init__(self, fname, dset, tstart=None, tend=None, cfg=None):
        if h5py.is_hdf5(fname):
            t, data = self._load_hdf5(fname, dset)
        else:
            t, data = self._load_csv(fname)

        # A config given explicitly takes precedence over any embedded one
        if cfg is not None:
            self.cfg = Inifile.load(cfg)

        # Sort by time and drop duplicates, keeping the latest samples
        idx = np.argsort(t, kind='stable')
        idx = idx[np.concatenate((np.diff(t[idx]) > 0, [True]))]
        t, data = t[idx], data[idx]

        # Restrict to the requested time window
        mask = np.ones(len(t), dtype=bool)
        if tstart is not None:
            mask &= t >= tstart
        if tend is not None:
            mask &= t <= tend

        self.t = t[mask]
        self.data = np.ascontiguousarray(data[mask].transpose(2, 1, 0))

    def _load_hdf5(self, fname, dset):
        with h5py.File(fname, 'r', swmr=True, libver='latest') as f:
            d = f[dset][:]
            attrs = f[dset].attrs

            self.fields = attrs['fields'].split(',')
            self.pts = attrs.get('pts', np.zeros((1, 3)))
            self.cfg = Inifile(attrs['config'])

            return d['t'], d['samples']

    def _load_csv(self, fname):
        with open(fname) as f:
            hdr, body = f.readline().strip(), f.read()

        dialect = csv.Sniffer().sniff(hdr)
        hdr = hdr.split(dialect.delimiter)
        raw = np.loadtxt(io.StringIO(body), delimiter=dialect.delimiter)

        self.cfg = None

        # Point series have coordinate columns; scalar histories do not
        if nd := len([c for c in hdr[1:4] if c in 'xyz']):
            coords = raw[:, 1:1 + nd]
            npts = len(np.unique(coords, axis=0))

            self.pts = coords[:npts]
            self.fields = hdr[1 + nd:]

            t = raw[::npts, 0]
            data = raw[:, 1 + nd:].reshape(len(t), npts, -1)
        else:
            self.pts = np.zeros((1, 3))
            self.fields = hdr[1:]

            t = raw[:, 0]
            data = raw[:, 1:].reshape(len(t), 1, -1)

        return t, data

    @property
    def elementscls(self):
        return get_elementscls(self.cfg)

    @property
    def namespace(self):
        # Columns as (npts, nt) arrays keyed by field name
        return dict(zip(self.fields, self.data))

    def columns(self, spec):
        if spec is None:
            return list(self.fields), self.data

        # Map any column indices through the recorded layout
        names = _split_spec(spec) if isinstance(spec, str) else list(spec)
        names = [self.fields[int(n)] if n.isdigit() else n for n in names]

        # Split unrecorded names into table quantities and expressions
        isid = lambda n: re.fullmatch(r'[^\W\d][\w-]*', n)
        need = [n for n in dict.fromkeys(names) if n not in self.fields]
        exprs = [n for n in need if not isid(n)]

        # Table-derived requests plus any expression dependencies
        tabs = [n for n in need if isid(n)]
        tabs += [v for e in exprs for v in expr_vars(e)]
        tabs = [n for n in dict.fromkeys(tabs) if n not in self.fields]

        # Evaluate the table-derived quantities over the recorded columns
        ns = self.namespace
        if tabs:
            ndims = self.pts.shape[1]
            wanted, support = soln_exprs(self.elementscls, ndims, self.cfg,
                                         tabs)
            ns |= eval_algebraic(wanted, ns, support)

        # Evaluate the inline expressions over the augmented namespace
        shape = self.data.shape[1:]
        ns |= {e: np.broadcast_to(npeval(e, ns), shape) for e in exprs}

        return names, np.stack([ns[n] for n in names])


def _resolve_pts(npts, spec):
    if spec is None:
        return list(range(npts))
    else:
        return [int(s) for s in spec.split(',')]


def _split_spec(spec):
    # Split a comma separated spec at zero parenthesis depth
    out, cur, depth = [], [], 0
    for c in spec:
        if c == ',' and not depth:
            out.append(''.join(cur).strip())
            cur = []
        else:
            depth += (c == '(') - (c == ')')
            cur.append(c)

    return [*out, ''.join(cur).strip()]


def _parse_signal(spec):
    field, pt = re.fullmatch(r'\s*(.+?)\s*(?:@\s*(\d+))?\s*', spec).groups()

    return field, int(pt or 0)


def _read_pts(path, ndims=None, skip=0):
    # Read the points
    with open(path) as f:
        pts = ''.join(list(f)[skip:])

    # Parse them
    dialect = csv.Sniffer().sniff(pts)
    pts = csv.reader(io.StringIO(pts), dialect=dialect)
    pts = np.array([[float(f) for f in p] for p in pts if p])

    # Validate the dimensionality
    if ndims and pts.shape[1] != ndims:
        raise ValueError('Invalid point set dimensionality')

    return pts


class SamplerCLIPlugin(BaseCLIPlugin):
    name = 'sampler'

    @classmethod
    def add_cli(cls, parser):
        sp = parser.add_subparsers()

        # Add command
        ap_add = sp.add_parser('add', help='sampler add --help')
        ap_add.add_argument('mesh', help='input mesh file')
        ap_add.add_argument('pts', help='input points file')
        ap_add.add_argument('-P', '--pname', help='partitioning to use')
        ap_add.add_argument('name', nargs='?', help='point set name')
        ap_add.add_argument('-f', '--force', action='count',
                            help='overwrite existing point set')
        ap_add.add_argument('--skip', type=int, default=0,
                            help='number of rows to skip')
        ap_add.set_defaults(process=cls.add_cmd)

        # List command
        ap_list = sp.add_parser('list', help='sampler list --help')
        ap_list.add_argument('mesh', help='input mesh file')
        ap_list.add_argument('-s', '--sep', default='\t', help='separator')
        ap_list.set_defaults(process=cls.list_cmd)

        # Dump command
        ap_dump = sp.add_parser('dump', help='sampler dump --help')
        ap_dump.add_argument('mesh', help='input mesh file')
        ap_dump.add_argument('name', help='point set')
        ap_dump.add_argument('-s', '--sep', default='\t', help='separator')
        ap_dump.set_defaults(process=cls.dump_cmd)

        # Remove command
        ap_remove = sp.add_parser('remove', help='sampler remove --help')
        ap_remove.add_argument('mesh', help='input mesh file')
        ap_remove.add_argument('name', help='point set')
        ap_remove.set_defaults(process=cls.remove_cmd)

        # Sample command
        ap_sample = sp.add_parser('sample', help='sampler sample --help')
        ap_sample.add_argument('mesh', help='input mesh file')
        ap_sample.add_argument('soln', help='input solution file')
        ap_sample.add_argument('-P', '--pname', help='partitioning to use')
        sample_opts = ap_sample.add_mutually_exclusive_group(required=True)
        sample_opts.add_argument('-n', '--name', help='point set')
        sample_opts.add_argument('-p', '--pts', help='input points file')
        ap_sample.add_argument('--skip', type=int, default=0,
                               help='number of rows to skip')
        ap_sample.add_argument(
            '-f', '--format',  choices=['conservative', 'primitive'],
             default='conservative', help='output format'
        )
        ap_sample.add_argument(
            '--postproc', dest='pp_plugins', action='append', default=[],
            metavar='PLUGIN', help='postprocessing plugin; may be repeated'
        )
        ap_sample.add_argument('--cfg', dest='pp_cfg',
                               help='config file for postproc plugins')
        ap_sample.add_argument('-s', '--sep', default='\t', help='separator')
        ap_sample.set_defaults(process=cls.sample_cmd)

        # Common time-series arguments
        def series_args(ap, fields=True):
            ap.add_argument('file', help='input HDF5 or CSV time series file')
            ap.add_argument('-d', '--dataset',
                            help='dataset name; required for HDF5 input')
            ap.add_argument('-c', '--cfg', dest='scfg',
                            help='config file; overrides any embedded one')
            if fields:
                ap.add_argument('-f', '--fields',
                                help='comma separated fields; defaults to '
                                     'all')
            ap.add_argument('-p', '--points',
                            help='comma separated point indices; defaults '
                                 'to all')
            ap.add_argument('--tstart', type=float, help='window start time')
            ap.add_argument('--tend', type=float, help='window end time')
            ap.add_argument('-s', '--sep', default='\t', help='separator')

        # Time-series statistics command
        ap_stats = sp.add_parser('stats', help='sampler stats --help')
        series_args(ap_stats)
        ap_stats.add_argument('--ci', action='store_true',
                              help='report autocorrelation-aware '
                                   'confidence intervals')
        ap_stats.add_argument('--level', type=float, default=0.68,
                              help='confidence level; defaults to 0.68')
        ap_stats.add_argument('--target', type=float,
                              help='relative mean error target; adds the '
                                   'averaging time required to attain it')
        ap_stats.add_argument('--maxlag', type=float,
                              help='maximum autocorrelation lag; defaults '
                                   'to a tenth of the series')
        ap_stats.set_defaults(process=cls.stats_cmd)

        # Initial-transient detection command
        ap_tr = sp.add_parser('transient', help='sampler transient --help')
        series_args(ap_tr)
        ap_tr.set_defaults(process=cls.transient_cmd)

        # Time-averaged statistics package command
        ap_tavg = sp.add_parser('tavg', help='sampler tavg --help')
        series_args(ap_tavg, fields=False)
        ap_tavg.add_argument('--stats', action='append', metavar='PKG',
                             required=True,
                             help='statistics package to evaluate over '
                                  'the series; may be repeated')
        ap_tavg.set_defaults(process=cls.tavg_cmd)

        # Power spectral density command
        ap_psd = sp.add_parser('psd', help='sampler psd --help')
        series_args(ap_psd)
        ap_psd.add_argument('-w', '--twin', type=float,
                            help='analysis window duration; defaults to '
                                 'an eighth of the series')
        ap_psd.add_argument('-o', '--overlap', type=float, default=0.5,
                            help='window overlap fraction')
        ap_psd.set_defaults(process=cls.psd_cmd)

        # Tone extraction command
        ap_tone = sp.add_parser('tone', help='sampler tone --help')
        series_args(ap_tone)
        ap_tone.add_argument('--fmin', type=float, required=True,
                             help='lower edge of the search band')
        ap_tone.add_argument('--fmax', type=float, required=True,
                             help='upper edge of the search band')
        ap_tone.set_defaults(process=cls.tone_cmd)

        # Autocorrelation command
        ap_ac = sp.add_parser('autocorr', help='sampler autocorr --help')
        series_args(ap_ac)
        ap_ac.add_argument('--maxlag', type=float,
                           help='maximum lag time; defaults to a tenth '
                                'of the series')
        ap_ac.add_argument('--nslots', type=int, default=200,
                           help='number of lag slots')
        ap_ac.set_defaults(process=cls.autocorr_cmd)

        # Two-point correlation command
        ap_tp = sp.add_parser('two-point', help='sampler two-point --help')
        series_args(ap_tp)
        ap_tp.add_argument('-r', '--ref', type=int, default=0,
                           help='reference point index')
        ap_tp.add_argument('--maxlag', type=float,
                           help='maximum lag time; defaults to a tenth '
                                'of the series')
        ap_tp.add_argument('--nslots', type=int, default=200,
                           help='number of lag slots')
        ap_tp.add_argument('--coherence', action='store_true',
                           help='coherence versus separation')
        ap_tp.add_argument('-w', '--twin', type=float,
                           help='analysis window duration; defaults to '
                                'an eighth of the series')
        ap_tp.add_argument('-o', '--overlap', type=float, default=0.5,
                           help='window overlap fraction')
        ap_tp.set_defaults(process=cls.twopoint_cmd)

        # Structure function command
        ap_sf = sp.add_parser('structfun', help='sampler structfun --help')
        series_args(ap_sf)
        ap_sf.add_argument('-n', '--orders', default='2,3',
                           help='comma separated orders; defaults to 2,3')
        ap_sf.add_argument('--maxlag', type=float,
                           help='maximum lag time; defaults to a tenth '
                                'of the series')
        ap_sf.add_argument('--nslots', type=int, default=200,
                           help='number of lag slots')
        ap_sf.add_argument('--signed', action='store_true',
                           help='use signed rather than absolute increments')
        ap_sf.set_defaults(process=cls.structfun_cmd)

        # Common signal-pair arguments
        def pair_args(ap):
            ap.add_argument('file', help='input HDF5 or CSV time series file')
            ap.add_argument('-d', '--dataset',
                            help='dataset name; required for HDF5 input')
            ap.add_argument('-c', '--cfg', dest='scfg',
                            help='config file; overrides any embedded one')
            ap.add_argument('-a', required=True, metavar='FIELD[@POINT]',
                            help='first signal')
            ap.add_argument('-b', required=True, metavar='FIELD[@POINT]',
                            help='second signal')
            ap.add_argument('--tstart', type=float, help='window start time')
            ap.add_argument('--tend', type=float, help='window end time')
            ap.add_argument('-s', '--sep', default='\t', help='separator')

        # Cross-correlation command
        ap_xc = sp.add_parser('xcorr', help='sampler xcorr --help')
        pair_args(ap_xc)
        ap_xc.add_argument('--maxlag', type=float,
                           help='maximum lag time; defaults to a tenth '
                                'of the series')
        ap_xc.add_argument('--nslots', type=int, default=200,
                           help='number of lag slots')
        ap_xc.set_defaults(process=cls.xcorr_cmd)

        # Coherence command
        ap_coh = sp.add_parser('coherence', help='sampler coherence --help')
        pair_args(ap_coh)
        ap_coh.add_argument('-w', '--twin', type=float,
                            help='analysis window duration; defaults to '
                                 'an eighth of the series')
        ap_coh.add_argument('-o', '--overlap', type=float, default=0.5,
                            help='window overlap fraction')
        ap_coh.set_defaults(process=cls.coherence_cmd)

    @cli_external
    def add_cmd(self, args):
        # Initialise MPI
        init_mpi()

        # Get our MPI info
        comm, rank, root = get_comm_rank_root()

        # Read the mesh
        reader = NativeReader(args.mesh, args.pname, construct_con=False)
        mesh = reader.mesh

        if rank == root:
            # Get the point set name
            pname = args.name or Path(args.pts).stem
            if not re.match(r'\w+$', pname):
                raise ValueError('Invalid point set name')

            # Check it does not already exist unless --force is given
            if f'plugins/sampler/{pname}' in mesh.raw and not args.force:
                raise ValueError(f'Point set {pname} already exists; use '
                                 '-f to replace')

            pts = _read_pts(args.pts, ndims=mesh.ndims, skip=args.skip)
        else:
            pts = None

        # Broadcast the points
        pts = comm.bcast(pts, root=root)

        # Identify which element each point is located in
        locs = PointLocator(mesh).locate(pts)

        # Close the mesh file so it can be reopened for writing
        reader.close()

        # Have the root rank write the point and location data out
        if rank == root:
            dtype = [('ploc', float, mesh.ndims), ('cidx', np.int16),
                     ('eidx', np.int64), ('tloc', float, mesh.ndims)]
            sinfo = np.empty(len(pts), dtype=dtype)
            sinfo['ploc'] = pts
            sinfo[['cidx', 'eidx', 'tloc']] = locs[['cidx', 'eidx', 'tloc']]

            with h5py.File(args.mesh, 'r+') as f:
                g = f.require_group('plugins/sampler')

                # Remove any existing sample point info
                if pname in g:
                    del g[pname]

                # Save the sample point info
                g[pname] = sinfo

    @cli_external
    def list_cmd(self, args):
        with h5py.File(args.mesh, 'r') as mesh:
            g = mesh.require_group('plugins/sampler')

            print('name', 'npts', sep=args.sep)
            for name, points in sorted(g.items()):
                print(name, len(points), sep=args.sep)

    @cli_external
    def dump_cmd(self, args):
        with h5py.File(args.mesh, 'r') as mesh:
            points = mesh[f'plugins/sampler/{args.name}']['ploc']
            ndim = points.shape[1]

            print(*'xyz'[:ndim], sep=args.sep)
            for p in points:
                print(*p, sep=args.sep)

    @cli_external
    def remove_cmd(self, args):
        with h5py.File(args.mesh, 'r+') as mesh:
            sgroup = mesh.get('plugins/sampler')

            if sgroup is None or args.name not in sgroup:
                raise ValueError(f'Point set {args.name} does not exist')

            del sgroup[args.name]

    @classmethod
    def _series_prep(cls, args):
        s = _Series(args.file, args.dataset, args.tstart, args.tend, args.scfg)
        fnames, data = s.columns(args.fields)
        pidxs = _resolve_pts(len(s.pts), args.points)

        return s.t, s.pts, fnames, pidxs, data

    @cli_external
    def stats_cmd(self, args):
        s = _Series(args.file, args.dataset, args.tstart, args.tend, args.scfg)
        pidxs = _resolve_pts(len(s.pts), args.points)
        fnames, data = s.columns(args.fields)

        d = data[:, pidxs].swapaxes(0, 1)

        # A required-time column is meaningless without the interval
        ci = args.ci or args.target is not None

        cols = ['point', 'field', 'mean', 'rms', 'min', 'max']
        if ci:
            cols += ['stderr', 'ci', 'tau-int', 'neff']
        if args.target is not None:
            cols += ['t-required']

        # Tones above this fraction of the variance inflate the interval
        fmin = 2 / (s.t[-1] - s.t[0])
        fmax = 0.25 / np.median(np.diff(s.t))

        print(*cols, sep=args.sep)
        for p, dp in zip(pidxs, d):
            for fn, x in zip(fnames, dp):
                # Time-weighted moments; robust to adaptive sampling
                m, var = tw_mean_var(s.t, x)
                row = [p, fn, m, np.sqrt(var), x.min(), x.max()]

                if ci:
                    m, se, hw, tau, neff = mean_ci(s.t, x, args.level,
                                                   args.maxlag)
                    row += [se, hw, tau, neff]

                    f0, a0 = tone(s.t, x, fmin, fmax)
                    if a0*a0 > 0.5*var:
                        print(f'# point {p} field {fn}: dominant tone at '
                              f'f = {f0:.5g} may inflate the interval',
                              file=sys.stderr)

                # Averaging time needed to reach the target mean error
                if args.target is not None:
                    z = hw / se
                    err = args.target*m
                    row += [2*tau*(z*np.sqrt(var)/err)**2]

                print(*row, sep=args.sep)

    @cli_external
    def transient_cmd(self, args):
        s = _Series(args.file, args.dataset, args.tstart, args.tend, args.scfg)
        pidxs = _resolve_pts(len(s.pts), args.points)
        fnames, data = s.columns(args.fields)

        d = data[:, pidxs].swapaxes(0, 1)

        # Breakdown to stderr; overall max alone to stdout for --tstart use
        tmax = s.t[0]
        print('point', 'field', 't-transient', sep=args.sep, file=sys.stderr)
        for p, dp in zip(pidxs, d):
            for fn, x in zip(fnames, dp):
                tt = s.t[mser(s.t, x)]
                tmax = max(tmax, tt)
                print(p, fn, tt, sep=args.sep, file=sys.stderr)

        print(tmax)

    @cli_external
    def tavg_cmd(self, args):
        s = _Series(args.file, args.dataset, args.tstart, args.tend, args.scfg)
        pidxs = _resolve_pts(len(s.pts), args.points)

        t, ns, ndims = s.t, s.namespace, s.pts.shape[1]

        avgs, derived, hidden, alg, _ = expand_exprs(
            s.cfg, ndims, s.elementscls, ', '.join(args.stats), {}, {}
        )

        # Time average each field integrand via the trapezoidal rule
        favg = {n: np.trapezoid(npeval(e, ns), t, axis=-1)/(t[-1] - t[0])
                for n, e in avgs.items()}

        # Finalise the algebraic derived quantities
        aderived = {n: e for n, e in derived.items() if n in alg}
        ahidden = {n: e for n, e in hidden.items() if n in alg}
        out = eval_algebraic(aderived, favg, ahidden)

        print('point', 'quantity', 'value', sep=args.sep)
        for p in pidxs:
            for n, v in out.items():
                print(p, n, v[p], sep=args.sep)

    @cli_external
    def psd_cmd(self, args):
        t, pts, fnames, pidxs, data = self._series_prep(args)

        if not 0 <= args.overlap < 1:
            raise ValueError('Overlap must be in [0, 1)')

        # Batch all requested channels through a single NUDFT pass
        x = data[:, pidxs].swapaxes(0, 1).reshape(-1, len(t)).T

        freqs, S = psd(t, x, args.twin, args.overlap)
        S = S.reshape(len(freqs), -1)

        names = [f'psd-{fn}-p{p}' for p in pidxs for fn in fnames]
        print('f', *names, sep=args.sep)
        for row in zip(freqs, *S.T):
            print(*row, sep=args.sep)

    @cli_external
    def tone_cmd(self, args):
        t, pts, fnames, pidxs, data = self._series_prep(args)

        print('point', 'field', 'f', 'amplitude', sep=args.sep)
        for p in pidxs:
            for fn, x in zip(fnames, data[:, p]):
                f, a = tone(t, x, args.fmin, args.fmax)
                print(p, fn, f, a, sep=args.sep)

    @cli_external
    def autocorr_cmd(self, args):
        t, pts, fnames, pidxs, data = self._series_prep(args)

        maxlag = args.maxlag or (t[-1] - t[0])/10

        cols, names, lags = [], [], None
        for p in pidxs:
            for fn, x in zip(fnames, data[:, p]):
                lags, r = acf(t, x, maxlag, args.nslots)

                # Integral time scale up to the first zero crossing
                zc = np.argmax(r < 0) or len(r)
                tint = np.trapezoid(r[:zc], lags[:zc])

                cols.append(r)
                names.append(f'r-{fn}-p{p}')
                print(f'# integral-time-{fn}-p{p} = {tint}')

        print('lag', *names, sep=args.sep)
        for row in zip(lags, *cols):
            print(*row, sep=args.sep)

    @cli_external
    def twopoint_cmd(self, args):
        t, pts, fnames, pidxs, data = self._series_prep(args)

        # Order the points by separation from the reference
        seps = np.linalg.norm(pts[pidxs] - pts[args.ref], axis=1)
        order = np.argsort(seps)
        pidxs, seps = np.asarray(pidxs)[order], seps[order]

        if args.coherence:
            self._twopoint_coh(args, t, seps, fnames, pidxs, data)
        else:
            self._twopoint_corr(args, t, seps, fnames, pidxs, data)

    def _twopoint_corr(self, args, t, seps, fnames, pidxs, data):
        maxlag = args.maxlag or (t[-1] - t[0])/10

        rows, r0s = [], []
        for p, sep in zip(pidxs, seps):
            row, r0 = [p, sep], []
            for dref, dp in data[:, [args.ref, p]]:
                lags, r = xcf(t, dref, dp, maxlag, args.nslots)
                k = np.abs(r).argmax()
                r0.append(r[len(lags)//2])
                row += [r0[-1], r[k], lags[k]]

            rows.append(row)
            r0s.append(r0)

        # Correlation length up to the first zero crossing
        for fn, r0 in zip(fnames, np.transpose(r0s)):
            zc = np.argmax(r0 < 0) or len(r0)
            clen = np.trapezoid(r0[:zc], seps[:zc])
            print(f'# corr-length-{fn} = {clen}')

        names = [f'{c}-{fn}' for fn in fnames
                 for c in ('r0', 'rpeak', 'lagpeak')]
        print('point', 'sep', *names, sep=args.sep)
        for row in rows:
            print(*row, sep=args.sep)

    def _twopoint_coh(self, args, t, seps, fnames, pidxs, data):
        if not 0 <= args.overlap < 1:
            raise ValueError('Overlap must be in [0, 1)')

        freqs, cols, names = None, [], []
        for fn, d in zip(fnames, data):
            coh = []
            for p in pidxs:
                x, y = d[args.ref], d[p]
                freqs, pxx, pyy, pxy, cnt = csd(t, x, y, args.twin,
                                                args.overlap)

                with np.errstate(divide='ignore', invalid='ignore'):
                    coh.append(np.abs(pxy)**2/(pxx*pyy))

            # Correct for the finite window-count coherence bias
            coh = np.clip((cnt*np.array(coh) - 1)/(cnt - 1), 0, None)

            cols.extend(coh)
            names.extend(f'coh-{fn}-p{p}' for p in pidxs)

            # Per-frequency e-folding length of an exponential decay fit
            good = coh > 1e-6
            with np.errstate(divide='ignore', invalid='ignore'):
                lnc = np.where(good, np.log(coh), 0)
                den = np.einsum('i,ij->j', seps**2, good)
                cols.append(-2*den/(seps @ lnc))

            names.append(f'declen-{fn}')

        print('f', *names, sep=args.sep)
        for row in zip(freqs, *cols):
            print(*row, sep=args.sep)

    @cli_external
    def structfun_cmd(self, args):
        t, pts, fnames, pidxs, data = self._series_prep(args)

        orders = [float(n) for n in args.orders.split(',')]
        maxlag = args.maxlag or (t[-1] - t[0])/10

        cols, names, lags = [], [], None
        for p in pidxs:
            for fn, x in zip(fnames, data[:, p]):
                lags, sf = structfun(t, x, orders, maxlag, args.nslots,
                                     args.signed)
                cols.extend(sf)
                names.extend(f's{n:g}-{fn}-p{p}' for n in orders)

        print('lag', *names, sep=args.sep)
        for row in zip(lags, *cols):
            print(*row, sep=args.sep)

    @classmethod
    def _pair_prep(cls, args):
        s = _Series(args.file, args.dataset, args.tstart, args.tend, args.scfg)

        (fa, pa), (fb, pb) = _parse_signal(args.a), _parse_signal(args.b)
        _, (ba, bb) = s.columns([fa, fb])

        return s.t, (ba[pa], bb[pb])

    @cli_external
    def xcorr_cmd(self, args):
        t, (x, y) = self._pair_prep(args)

        maxlag = args.maxlag or (t[-1] - t[0])/10
        lags, r = xcf(t, x, y, maxlag, args.nslots)

        # Positive lag means the first signal leads the second
        i = np.abs(r).argmax()
        print(f'# peak r = {r[i]} at lag = {lags[i]}')

        print('lag', 'r', sep=args.sep)
        for row in zip(lags, r):
            print(*row, sep=args.sep)

    @cli_external
    def coherence_cmd(self, args):
        t, (x, y) = self._pair_prep(args)

        if not 0 <= args.overlap < 1:
            raise ValueError('Overlap must be in [0, 1)')

        f, pxx, pyy, pxy, cnt = csd(t, x, y, args.twin, args.overlap)

        # Magnitude-squared coherence and cross-spectrum phase
        with np.errstate(divide='ignore', invalid='ignore'):
            coh = np.abs(pxy)**2/(pxx*pyy)

        phase = np.arctan2(pxy.imag, pxy.real)

        # Correct for the finite window-count coherence bias
        coh = np.clip((cnt*coh - 1)/(cnt - 1), 0, None)

        print('f', 'coherence', 'phase', sep=args.sep)
        for row in zip(f, coh, phase):
            print(*row, sep=args.sep)

    @cli_external
    def sample_cmd(self, args):
        # Initialise MPI
        init_mpi()

        # Get our MPI info
        comm, rank, root = get_comm_rank_root()

        # Read the mesh and solution
        reader = NativeReader(args.mesh, args.pname, construct_con=False)
        mesh, soln = reader.load_subset_mesh_soln(args.soln)

        # Dimension and field names
        dims = 'xyz'[:mesh.ndims]
        fields = soln.fields

        # Read the sample points from a CSV file
        if args.pts:
            if rank == root:
                pts = _read_pts(args.pts, ndims=mesh.ndims, skip=args.skip)
            else:
                pts = None

            pts = comm.bcast(pts, root=root)
            locs = None
        # Obtain the pre-processed sample points from the mesh
        else:
            if rank == root:
                pdata = mesh.raw[f'plugins/sampler/{args.name}'][:]
            else:
                pdata = None

            pdata = comm.bcast(pdata, root=root)

            pts = pdata['ploc']
            locs = pdata[['cidx', 'eidx', 'tloc']]

        prefix = soln.stats.get('data', 'prefix')

        # Postproc plugins on solution files require primitive format
        if args.pp_plugins and prefix == 'soln' and args.format != 'primitive':
            raise ValueError('Postproc plugins require --format=primitive')

        # Requested plugins plus file defaults (e.g. tavg statistics)
        names = list(args.pp_plugins)
        if soln.stats.hasopt('data', 'postproc'):
            names.extend(soln.stats.get('data', 'postproc').split())

        # Resolve the data source and its postproc pipeline
        source = get_source(prefix, soln.config, soln.stats, mesh.ndims)
        pp_cfg = Inifile.load(args.pp_cfg) if args.pp_cfg else soln.config
        pipe = source.pipeline(names, 'volume', pp_cfg)

        # Give the data source a chance to augment the solution
        nstored = len(soln.layout)
        source.prepare(mesh, soln, pipe.plugins)

        # Determine if gradient and residual data are present
        has_grads = bool(soln.grad_data)
        has_resid = bool(soln.resid_data)

        # If gradients or residuals exist, stack them into the solution
        sdata = []
        for etype in mesh.eidxs:
            d = soln.data[etype]
            if has_grads:
                g = soln.grad_data[etype].transpose(1, 2, 0, 3)
                g = g.reshape(g.shape[0], -1, g.shape[3])
                d = np.concatenate([d, g], axis=1)
            if has_resid:
                d = np.concatenate([d, soln.resid_data[etype]], axis=1)

            sdata.append(d)

        # Handle conversion from conservative to primitive variables
        if args.format == 'primitive':
            if prefix != 'soln':
                raise ValueError('Primitive output only supported for '
                                 'solution files')

            # Obtain the element class associated with the solution
            elementscls = get_elementscls(soln.config)
            vmap = elementscls.privars(mesh.ndims, soln.config)

            fields = list(vmap)
            if has_grads:
                fields.extend(f'grad_{v}_{d}' for v in vmap for d in dims)
            if has_resid:
                fields.extend(f'resid_{v}' for v in vmap)

            process = partial(con_block_to_pri, elementscls, soln.config,
                              mesh.ndims, grads=has_grads, resid=has_resid)
        else:
            process = None
            fields = list(soln.layout)
            if has_grads:
                fields.extend(f'grad_{v}_{d}'
                              for v in soln.fields for d in dims)
            if has_resid:
                fields.extend(f'resid_{v}' for v in soln.fields)

        # Construct and configure the point sampler
        sampler = PointSampler(mesh, pts, locs)
        sampler.configure_with_cfg_nvars(soln.config, len(fields))

        # Sample the solution
        samps = sampler.sample(sdata, process=process)

        # Have the root rank post-process and write the samples
        if rank == root:
            # Run any requested post-processing plugins
            if pipe.plugins:
                # Adapt the raw samples for any post-processing plugins
                out = pipe(soln, samps.T, pts.T)

                # Drop any working rows appended by the data source
                if nextra := len(soln.layout) - nstored:
                    fields = fields[:-nextra]
                    samps = samps[:, :len(fields)]

                extra_cols = []
                for name, arr in out.items():
                    if name.startswith('_') or name in fields:
                        continue

                    fields.extend(pipe.fields[name])
                    extra_cols.append(np.atleast_2d(arr.T).T)

                samps = np.concatenate([samps, *extra_cols], axis=1)

            # Write out the header
            print(*dims, *fields, sep=args.sep)

            # Write out the samples
            for ploc, samp in zip(pts, samps):
                print(*ploc, *samp, sep=args.sep)
