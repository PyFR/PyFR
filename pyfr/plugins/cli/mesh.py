from collections import namedtuple
import json
import shutil

import numpy as np

from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root, init_mpi, mpi, scal_coll
from pyfr.plugins.base import BaseCLIPlugin
from pyfr.plugins.common import cli_external
from pyfr.quality import element_metrics, neighbour_size_ratio
from pyfr.readers.native import NativeReader
from pyfr.shapes import BaseShape
from pyfr.util import subclass_where, subclasses, tty


EleInfo = namedtuple(
    'EleInfo', 'neles nupts nmpts sptsord curved eidxs metrics n_inverted '
    'n_nan n_poor_scaled_jac n_high_aspect max_nsr', defaults=(None,)
)
FieldStats = namedtuple('FieldStats',
                        'n min max mean std hist_counts hist_edges')
WorstEle = namedtuple('WorstEle', 'etype eidx val ploc')


def _reduce_field(arr, nbins=10):
    comm, _, _ = get_comm_rank_root()

    v = np.asanyarray(arr, dtype=float).ravel()
    v = v[np.isfinite(v)]

    lmin = np.min(v) if len(v) else np.inf
    lmax = np.max(v) if len(v) else -np.inf

    # Reduce the extrema over all ranks to fix the bin edges
    lo = scal_coll(comm.Allreduce, lmin, op=mpi.MIN)
    hi = scal_coll(comm.Allreduce, lmax, op=mpi.MAX)

    n = scal_coll(comm.Allreduce, len(v))
    s = scal_coll(comm.Allreduce, np.sum(v))
    ss = scal_coll(comm.Allreduce, np.sum(v*v))

    # Bin the values when they span a range
    if lo < hi:
        edges = np.linspace(lo, hi, nbins + 1)
        counts = np.histogram(v, bins=edges)[0]
        comm.Allreduce(mpi.IN_PLACE, counts)
    else:
        edges = np.zeros(nbins + 1)
        counts = np.zeros(nbins, dtype=int)

    mean = s / n if n else 0.0
    std = np.sqrt(max(ss / n - mean**2, 0)) if n else 0.0

    return FieldStats(n, lo, hi, mean, std, counts, edges)


def _find_worst(etype, arr, ploc, eidxs, n=10, minimise=True):
    # Rank NaNs as the worst elements of all
    fill = -np.inf if minimise else np.inf
    val = np.where(np.isnan(arr), fill, arr)

    # Break ties on the element index
    idxs = np.lexsort((eidxs, val if minimise else -val))[:n]

    # Locate each element by the centroid of its solution points
    cents = np.mean(ploc[..., idxs], axis=0).T

    return [WorstEle(etype, ei, v, c)
            for ei, v, c in zip(eidxs[idxs], arr[idxs], cents)]


def _highlight(s, on):
    return f'{tty.red}{tty.bold}{s}{tty.reset}' if on else s


def _print_worst_table(worst, title, col):
    t = tty
    print()
    print(f'{t.bold}{title}:{t.reset}')
    print(f'  {'Type':<8} {'Element':>10} {col:>12} Location')
    print(f'  {'-'*8} {'-'*10} {'-'*12} {'-'*20}')

    for j, (etype, el, val, loc) in enumerate(worst):
        loc_str = ', '.join(f'{c:.3f}' for c in loc)

        # Call out the single worst element in the table
        print(_highlight(f'  {etype:<8} {el:>10} {val:>12.4g} ({loc_str})',
                         j == 0))


def _print_histogram(fs, title, highlight_min=False, width=30):
    counts, edges = fs.hist_counts, fs.hist_edges
    total, span = counts.sum(), np.ptp(edges)

    # Skip when empty or the range is negligible relative to magnitude
    if not total or span < 1e-6*max(abs(edges[-1]), 1e-10):
        return

    max_count = counts.max() or 1

    # Set the printed precision from the span
    exp = int(np.floor(np.log10(span)))
    if -6 <= exp <= 4:
        fmt = f'10.{max(0, 3 - exp)}f'
    else:
        fmt = '10.3e'

    print(f'  {tty.cyan}{title} Distribution:{tty.reset}')

    # Unicode block characters for smooth bars
    blocks = ' ▏▎▍▌▋▊▉█'

    for idx, (lo, hi, c) in enumerate(zip(edges[:-1], edges[1:], counts)):
        frac = c / max_count * width
        full = int(frac)
        part = int(8*(frac - full))
        bar = '█'*full + (blocks[part] if part else '')
        bar = bar.ljust(width)
        pct = 100 * c / total
        line = f'  [{lo:{fmt}},{hi:{fmt}}) │{bar}│ {c:>5} ({pct:4.1f}%)'
        print(_highlight(line, highlight_min and idx == 0 and c > 0))

    print()


def _format_stats(fs, name, highlight_min=False, highlight_max=False):
    if not fs.n:
        return f'  {name}: No valid data'

    return (f'  {name}:\n'
            f'    Min: {_highlight(f'{fs.min:9.3g}', highlight_min)}  '
            f'Max: {_highlight(f'{fs.max:9.3g}', highlight_max)}  '
            f'Mean: {fs.mean:9.3g} ± {fs.std:9.3g}')


class _MeshAnalyser:
    # Per-element-type quantities and counts
    statlabels = {'scaled_jac': 'Scaled Jacobian', 'h': 'Mesh Scale (h)',
                  'aspect': 'Aspect Ratio', 'vol': 'Element Volume',
                  'jvar': 'Jacobian Variation',
                  'curved': 'Scaled Jacobian (curved)'}
    sumcounts = ('neles', 'ncurved', 'n_inverted', 'n_nan', 'n_poor')
    maxcounts = ('nupts', 'nmpts', 'q')

    # Whole-mesh counts and minimums
    gsums = ('n_total', 'n_inverted', 'n_nan', 'n_poor_scaled_jac',
             'n_high_aspect', 'n_curved')
    gmins = ('min_scaled_jac', 'min_curved_scaled_jac')

    def __init__(self, mesh, elementscls, basismap, cfg, jac_thresh,
                 ar_thresh, sr_thresh=0):
        self.mesh = mesh
        self.cfg = cfg
        self.order = cfg.getint('solver', 'order')
        self.jac_thresh = jac_thresh
        self.ar_thresh = ar_thresh
        self.nsr_thresh = sr_thresh
        self.nsr = None

        self.etypes = {}
        self.stats = {k: 0 for k in self.gsums}
        self.stats |= {k: np.inf for k in self.gmins}

        for etype, spts in mesh.spts.items():
            ele = elementscls(basismap[etype], spts, cfg)
            m = element_metrics(ele)
            curved = mesh.spts_curved[etype]
            n_curved = int(np.sum(curved))

            # Count issues per element (int cast for JSON serialization)
            nan = np.any(np.isnan(m.djac), axis=0)
            n_inv = int(np.sum((m.scaled_jac <= 0) & ~nan))
            n_nan = int(np.sum(nan))
            n_pj = int(np.sum(m.scaled_jac < jac_thresh))
            n_ha = int(np.sum(np.any(m.aspect > ar_thresh, axis=0)))

            self.etypes[etype] = EleInfo(
                ele.neles, ele.nupts, ele.nmpts, ele.basis.nsptsord, curved,
                mesh.eidxs[etype], m, n_inv, n_nan, n_pj, n_ha
            )

            counts = {'n_total': ele.neles, 'n_inverted': n_inv,
                      'n_nan': n_nan, 'n_poor_scaled_jac': n_pj,
                      'n_high_aspect': n_ha, 'n_curved': n_curved}

            for k in self.gsums:
                self.stats[k] += counts[k]

            # Track global minimums
            mins = {'min_scaled_jac': np.nanmin(m.scaled_jac)}
            if n_curved:
                csj = np.nanmin(m.scaled_jac[curved])
                mins['min_curved_scaled_jac'] = csj

            for k, v in mins.items():
                self.stats[k] = min(self.stats[k], v)

        if sr_thresh > 0:
            self._compute_nsr()

    def _compute_nsr(self):
        comm, _, _ = get_comm_rank_root()

        # Worst neighbour size ratio seen by each element
        cl = {et: r.metrics.char_len for et, r in self.etypes.items()}
        nsr = neighbour_size_ratio(self.mesh, cl)

        # Stash the per-element ratio for display
        for et in nsr:
            self.etypes[et] = self.etypes[et]._replace(max_nsr=nsr[et])

        # Neighbour ratio statistics and the count above the threshold
        local = np.concatenate(list(nsr.values()))
        nhigh = int(np.sum(local > self.nsr_thresh))

        self.nsr = _reduce_field(local)
        self.stats['n_high_nsr'] = scal_coll(comm.Allreduce, nhigh)

    def _gather_worst(self, get_arr, n, minimise=True):
        comm, rank, root = get_comm_rank_root()

        # For each element type, find the worst n elements
        candidates = []
        for etype, res in self.etypes.items():
            candidates += _find_worst(etype, get_arr(res), res.metrics.ploc,
                                      res.eidxs, n=n, minimise=minimise)

        def key(c):
            val = c.val if minimise else -c.val
            return (val, c.etype, c.eidx)

        # Locally sort the candidates
        candidates.sort(key=key)

        # Gather the top n candidates from each rank to the root rank
        candidates = comm.gather(candidates[:n], root=root)
        if rank == root:
            candidates = [c for cl in candidates for c in cl]
            candidates.sort(key=key)
            return candidates[:n]
        else:
            return []

    def _highlight_min(self, etype, f):
        fs = self.fieldstats[etype, f]

        if f in ('scaled_jac', 'curved'):
            return fs.min < self.jac_thresh
        elif f == 'vol':
            return fs.min <= 0
        elif f == 'h':
            mh = self.min_h_ele
            return mh is not None and etype == mh.etype
        else:
            return False

    def _local_fields(self, etype):
        res = self.etypes.get(etype)

        # Return empty arrays for types absent from this rank
        if res is None:
            return dict.fromkeys(self.statlabels, np.empty(0))
        else:
            m = res.metrics
            return {'scaled_jac': m.scaled_jac, 'h': np.min(m.h, axis=0),
                    'aspect': np.max(m.aspect, axis=0), 'vol': m.vol,
                    'jvar': m.jvar, 'curved': m.scaled_jac[res.curved]}

    def _local_counts(self, etype):
        res = self.etypes.get(etype)

        if res is None:
            return dict.fromkeys(self.sumcounts + self.maxcounts, 0)
        else:
            return {'neles': res.neles, 'ncurved': int(res.curved.sum()),
                    'n_inverted': res.n_inverted, 'n_nan': res.n_nan,
                    'n_poor': res.n_poor_scaled_jac, 'nupts': res.nupts,
                    'nmpts': res.nmpts, 'q': res.sptsord}

    def reduce(self, n_worst=0):
        # Reduce per-rank stats to root via MPI
        comm, rank, root = get_comm_rank_root()

        for key in self.gsums:
            self.stats[key] = scal_coll(comm.Allreduce, self.stats[key])

        for key in self.gmins:
            self.stats[key] = scal_coll(comm.Allreduce, self.stats[key],
                                        op=mpi.MIN)

        self.ginfo, self.fieldstats = {}, {}

        for etype in self.mesh.etypes:
            for f, arr in self._local_fields(etype).items():
                self.fieldstats[etype, f] = _reduce_field(arr)

            # Sum the element counts and take the sizes from any holder
            loc = self._local_counts(etype)
            gi = {k: scal_coll(comm.Allreduce, loc[k])
                  for k in self.sumcounts}
            gi |= {k: scal_coll(comm.Allreduce, loc[k], op=mpi.MAX)
                   for k in self.maxcounts}

            self.ginfo[etype] = gi

        # Gather worst-N candidates from all ranks
        self.worst_sj = self._gather_worst(lambda r: r.metrics.scaled_jac,
                                           n_worst)

        # Take the CFL limiting element from the worst mesh scales
        worst_h = self._gather_worst(lambda r: np.min(r.metrics.h, axis=0),
                                     max(n_worst, 1))
        self.worst_h = worst_h[:n_worst]
        self.min_h_ele = worst_h[0] if worst_h else None

        if self.nsr_thresh > 0:
            self.worst_nsr = self._gather_worst(lambda r: r.max_nsr, n_worst,
                                                minimise=False)
        else:
            self.worst_nsr = []

        return rank == root

    def _print_etype(self, etype):
        t, gi = tty, self.ginfo[etype]

        print(f'{t.bold}Element Type: {etype}{t.reset} '
              f'({gi['neles']} elements, {gi['ncurved']} curved), '
              f'p = {self.order}, q = {gi['q']}, '
              f'nupts = {gi['nupts']}, nmpts = {gi['nmpts']}\n')

        for f, label in self.statlabels.items():
            if f != 'curved' or gi['ncurved']:
                hl = self._highlight_min(etype, f)
                print(_format_stats(self.fieldstats[etype, f], label,
                                    highlight_min=hl), '', sep='\n')

        # Scaled Jacobian stats filtered to curved elements
        if gi['ncurved']:
            _print_histogram(self.fieldstats[etype, 'curved'],
                             self.statlabels['curved'],
                             self._highlight_min(etype, 'curved'))

        _print_histogram(self.fieldstats[etype, 'h'], 'Mesh Scale',
                         self._highlight_min(etype, 'h'))

    def _print_nsr(self):
        hl = self.nsr.max > self.nsr_thresh

        print(_format_stats(self.nsr, 'Neighbour Size Ratio',
                            highlight_max=hl), '', sep='\n')
        _print_histogram(self.nsr, 'Neighbour Size Ratio')

    def _print_summary(self, w):
        t, s = tty, self.stats

        print('-'*w, f'{t.bold}Summary{t.reset}', '-'*w, sep='\n')

        jl = f'Scaled Jacobian < {self.jac_thresh}:'
        al = f'Aspect ratio > {self.ar_thresh}:'
        counts = {'Inverted elements (J ≤ 0):': s['n_inverted'],
                  'Elements with NaN Jacobian:': s['n_nan'],
                  jl: s['n_poor_scaled_jac'], al: s['n_high_aspect']}

        if self.nsr:
            sl = f'Neighbour size ratio > {self.nsr_thresh}:'
            counts[sl] = s['n_high_nsr']

        # Size the label column so that long thresholds still line up
        lw = max(len(l) for l in counts)

        print(f'  {'Curved elements:'.ljust(lw)}  '
              f'{s['n_curved']} / {s['n_total']}')

        for label, val in counts.items():
            c = t.green if val == 0 else t.red
            print(f'  {label.ljust(lw)}  {c}{val}{t.reset}')

        print()

        min_csj = s['min_curved_scaled_jac']
        if np.isfinite(min_csj):
            c = t.red if min_csj < self.jac_thresh else t.cyan
            print(f'  {c}Min scaled Jacobian (curved):{t.reset} '
                  f'{min_csj:9.3g}')

        mh = self.min_h_ele
        if mh is not None and np.isfinite(mh.val):
            dt_factor = mh.val / (2*self.order + 1)
            print(f'  {t.cyan}Likely CFL limiting element:{t.reset}'
                  f' {mh.etype} {mh.eidx} (h = {mh.val:.3g})')
            print(f'  {t.cyan}Geometric dt factor:{t.reset} '
                  f'h_min/(2p+1) = {dt_factor:9.3g}')

        if self.nsr:
            print(f'  {t.cyan}Max neighbour size ratio:{t.reset} '
                  f'{self.nsr.max:9.3g}')

    def _print_worst(self, w):
        print('='*w)

        tables = [
            (self.worst_sj, 'Worst Elements by Scaled Jacobian', 'Scaled J'),
            (self.worst_h, 'Smallest Mesh Scale (CFL limiting)', 'h_min'),
            (self.worst_nsr, 'Worst Neighbour Size Ratios', 'NSR'),
        ]

        for worst, title, col in [t for t in tables if t[0]]:
            _print_worst_table(worst, title, col)

    def output_text(self, n_worst):
        t, w = tty, min(shutil.get_terminal_size().columns, 72)

        print(f'{t.bold}Mesh Quality Report{t.reset}', '='*w, '', sep='\n')

        for etype in self.mesh.etypes:
            self._print_etype(etype)

        if self.nsr:
            self._print_nsr()

        self._print_summary(w)

        if n_worst > 0:
            self._print_worst(w)

    def output_json(self):
        s = self.stats
        sr = self.nsr

        # Determine status
        high_nsr = sr is not None and s['n_high_nsr']
        if s['n_inverted'] or s['n_nan']:
            status = 'error'
        elif s['n_poor_scaled_jac'] or s['n_high_aspect'] or high_nsr:
            status = 'warning'
        else:
            status = 'ok'

        # Extract global metrics, converting inf to None for JSON
        mh = self.min_h_ele.val if self.min_h_ele is not None else np.inf
        glob = {k: s[k] for k in self.gsums if k != 'n_total'}
        for k in self.gmins:
            glob[k] = float(s[k]) if np.isfinite(s[k]) else None

        if np.isfinite(mh):
            glob['min_h'] = float(mh)
            glob['geometric_dt_factor'] = mh / (2*self.order + 1)
        else:
            glob['min_h'] = None
            glob['geometric_dt_factor'] = None

        if sr:
            glob['n_high_size_ratio'] = s['n_high_nsr']
            glob['max_size_ratio'] = sr.max

        output = {
            'status': status, 'order': self.order,
            'element_types': {}, 'global': glob,
        }

        for etype in self.mesh.etypes:
            gi = self.ginfo[etype]
            etd = {'neles': gi['neles'], 'n_curved': gi['ncurved'],
                   'nupts': gi['nupts'], 'nmpts': gi['nmpts'],
                   'mesh_order': gi['q'], 'n_inverted': gi['n_inverted'],
                   'n_nan': gi['n_nan'], 'n_poor_scaled_jac': gi['n_poor']}

            for key in self.statlabels:
                fs = self.fieldstats[etype, key]
                if fs.n:
                    etd[f'min_{key}'] = float(fs.min)
                    etd[f'max_{key}'] = float(fs.max)
                    etd[f'mean_{key}'] = float(fs.mean)

            output['element_types'][etype] = etd

        print(json.dumps(output, indent=2))


class MeshCLIPlugin(BaseCLIPlugin):
    name = 'mesh'

    @classmethod
    def add_cli(cls, parser):
        parser.add_argument('mesh', help='mesh file')
        parser.add_argument('cfg', help='config file')
        parser.add_argument('-P', '--pname', help='partitioning to use')
        parser.add_argument('--json', action='store_true',
                            help='output as JSON')
        parser.add_argument('--worst', type=int, default=0, metavar='N',
                            help='show N worst elements')
        parser.add_argument('--order', type=int, metavar='P',
                            help='override polynomial order from config')
        parser.add_argument('--jac-thresh', type=float, default=0.5,
                            metavar='J', help='scaled Jacobian threshold')
        parser.add_argument('--ar-thresh', type=float, default=20,
                            metavar='AR', help='aspect ratio threshold')
        parser.add_argument('--nsr-thresh', type=float, default=5,
                            metavar='NSR',
                            help='neighbour size ratio threshold')

        parser.set_defaults(process=cls.process_cmd)

    @cli_external
    def process_cmd(self, args):
        from pyfr.solvers.base import BaseSystem

        init_mpi()

        reader = NativeReader(args.mesh, pname=args.pname,
                              construct_con=args.nsr_thresh > 0)
        cfg = Inifile.load(args.cfg)

        # Override polynomial order from config if provided
        if args.order is not None:
            cfg.set('solver', 'order', args.order)

        basismap = {b.name: b for b in subclasses(BaseShape, just_leaf=True)}
        system = subclass_where(BaseSystem, name=cfg.get('solver', 'system'))

        ma = _MeshAnalyser(reader.mesh, system.elementscls, basismap, cfg,
                           args.jac_thresh, args.ar_thresh, args.nsr_thresh)

        if ma.reduce(args.worst):
            if args.json:
                ma.output_json()
            else:
                ma.output_text(args.worst)
