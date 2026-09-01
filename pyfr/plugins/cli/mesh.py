from collections import namedtuple
import json
from pathlib import Path
import shutil

import numpy as np

from pyfr.inifile import Inifile
from pyfr.mpiutil import (autofree, get_comm_rank_root, init_mpi, mpi,
                          scal_coll)
from pyfr.plugins.base import BaseCLIPlugin
from pyfr.plugins.common import cli_external
from pyfr.readers.native import NativeReader
from pyfr.shapes import BaseShape
from pyfr.util import subclass_where, subclasses, tty
from pyfr.writers.native import NativeWriter


Metrics = namedtuple('Metrics',
                     'djac h aspect scaled_jac jvar vol char_len ploc')
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


def _scaled_jac(ele):
    sj = None

    for name in ('upts', 'fpts'):
        jac = ele.jac_at_np(name)

        # Normalise the determinant by the tangent vector norms
        tnorm = np.prod(np.linalg.norm(jac, axis=-1), axis=-1)
        with np.errstate(divide='ignore', invalid='ignore'):
            s = np.where(tnorm > 0, np.linalg.det(jac) / tnorm, 0)

        # Keep the worst value seen at any of the solver's points
        s = np.min(s, axis=0)
        sj = s if sj is None else np.minimum(sj, s)

    return sj


def _compute_metrics(ele):
    # Jacobian determinant at solution points
    djac = np.linalg.det(ele.jac_at_np('upts'))
    with np.errstate(divide='ignore', invalid='ignore'):
        rcpdjac = 1.0 / djac

    # Metric terms at solution points
    smats = ele.smat_at_np('upts')

    # J^{-1} scaled by 1/det(J)
    jinv = smats * rcpdjac[None, :, None, :]

    # Mesh scale: h_i = 2 / ||J^{-1}_i||_2 per reference direction
    h_per_dir = 2.0 / np.sqrt(np.sum(jinv**2, axis=2))
    h_min = np.min(h_per_dir, axis=0)
    h_max = np.max(h_per_dir, axis=0)

    # Aspect ratio
    aspect = h_max / h_min

    # Scaled Jacobian at the solution and flux points
    scaled_jac = _scaled_jac(ele)

    # Variation of the Jacobian determinant within each element
    adjac = np.abs(djac)
    with np.errstate(divide='ignore', invalid='ignore'):
        jvar = np.min(adjac, axis=0) / np.max(adjac, axis=0)

    # Element volume from the solution point quadrature
    wts = ele.basis.ubasis.invvdm[:, 0]
    vol = wts.sum() * (wts @ djac)

    # Volume-based characteristic length
    char_len = np.abs(np.mean(djac, axis=0))**(1.0 / ele.ndims)

    # Physical locations
    ploc = ele.ploc_at_np('upts')

    return Metrics(djac, h_min, aspect, scaled_jac, jvar, vol, char_len, ploc)


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


def _print_worst_table(worst, title, col):
    t = tty
    print()
    print(f'{t.bold}{title}:{t.reset}')
    print(f'  {'Type':<8} {'Element':>10} {col:>12} Location')
    print(f'  {'-'*8} {'-'*10} {'-'*12} {'-'*20}')

    for j, (etype, el, val, loc) in enumerate(worst):
        loc_str = ', '.join(f'{c:.3f}' for c in loc)

        if j == 0:
            print(f'{t.red}{t.bold}  {etype:<8} {el:>10} {val:>12.4g} '
                  f'({loc_str}){t.reset}')
        else:
            print(f'  {etype:<8} {el:>10} {val:>12.4g} ({loc_str})')


def _render_histogram(counts, edges, width=30, highlight_min=False):
    t = tty
    total = counts.sum()

    if not total:
        return []

    # Skip if range is negligible relative to magnitude
    span = np.ptp(edges)
    if span < 1e-6*max(abs(edges[-1]), 1e-10):
        return []

    max_count = counts.max() or 1

    # Set the printed precision from the span
    exp = int(np.floor(np.log10(span)))
    if -6 <= exp <= 4:
        fmt = f'10.{max(0, 3 - exp)}f'
    else:
        fmt = '10.3e'

    # Unicode block characters for smooth bars
    blocks = ' ▏▎▍▌▋▊▉█'
    lines = []

    for idx, (lo, hi, c) in enumerate(zip(edges[:-1], edges[1:], counts)):
        frac = c / max_count * width
        full = int(frac)
        part = int(8*(frac - full))
        bar = '█'*full + (blocks[part] if part else '')
        bar = bar.ljust(width)
        pct = 100 * c / total
        line = f'  [{lo:{fmt}},{hi:{fmt}}) │{bar}│ {c:>5} ({pct:4.1f}%)'
        if highlight_min and idx == 0 and c > 0:
            line = f'{t.red}{t.bold}{line}{t.reset}'
        lines.append(line)

    return lines


def _format_stats(fs, name, highlight_min=False):
    t = tty

    if not fs.n:
        return f'  {name}: No valid data'

    if highlight_min:
        mn_str = f'{t.red}{t.bold}{fs.min:9.3g}{t.reset}'
    else:
        mn_str = f'{fs.min:9.3g}'

    return (f'  {name}:\n'
            f'    Min: {mn_str}  Max: {fs.max:9.3g}  '
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
        self.stats |= {'min_h': np.inf, 'min_h_etype': None,
                       'min_h_eidx': None}

        for etype, spts in mesh.spts.items():
            ele = elementscls(basismap[etype], spts, cfg)
            m = _compute_metrics(ele)
            curved = mesh.spts_curved[etype]
            n_curved = int(np.sum(curved))

            # Count issues per element (int cast for JSON serialization)
            n_inv = int(np.sum(np.any(m.djac <= 0, axis=0)))
            n_nan = int(np.sum(np.any(np.isnan(m.djac), axis=0)))
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
            sjmin = np.nanmin(m.scaled_jac)
            self.stats['min_scaled_jac'] = min(self.stats['min_scaled_jac'],
                                               sjmin)

            if n_curved:
                csjmin = np.nanmin(m.scaled_jac[curved])
                k = 'min_curved_scaled_jac'
                self.stats[k] = min(self.stats[k], csjmin)

            h_min_per_ele = np.nanmin(m.h, axis=0)
            min_h_idx = np.nanargmin(h_min_per_ele)
            min_h_val = h_min_per_ele[min_h_idx]
            if min_h_val < self.stats['min_h']:
                self.stats['min_h'] = min_h_val
                self.stats['min_h_etype'] = etype
                self.stats['min_h_eidx'] = mesh.eidxs[etype][min_h_idx]

        if sr_thresh > 0:
            self._compute_nsr()

    def _compute_nsr(self):
        mesh = self.mesh
        comm, _, _ = get_comm_rank_root()

        # Per-element characteristic length and worst neighbour ratio
        cl = {et: r.metrics.char_len for et, r in self.etypes.items()}
        nsr = {et: np.ones(r.neles) for et, r in self.etypes.items()}

        def process(con, lcl, rcl):
            face_r = np.maximum(lcl, rcl) / np.minimum(lcl, rcl)
            for et, _, ei, mask in con.foreach():
                np.maximum.at(nsr[et], ei, face_r[mask])

        # Internal faces
        if mesh.con:
            for lhs, rhs in [mesh.con, mesh.con[::-1]]:
                process(lhs, lhs.map_eles(cl), rhs.map_eles(cl))

        # MPI faces
        nbrs = sorted(mesh.con_p)
        ncomm = autofree(comm.Create_dist_graph_adjacent(nbrs, nbrs))
        send = [con.map_eles(cl) for con in mesh.con_p.values()]
        recv = ncomm.neighbor_alltoall(send)
        for con, rcl in zip(mesh.con_p.values(), recv):
            process(con, con.map_eles(cl), rcl)

        # Stash per-element max_nsr for export and display
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
            return etype == self.stats['min_h_etype']
        else:
            return False

    def _print_histogram(self, fs, title, highlight_min=False):
        t = tty
        hist = _render_histogram(fs.hist_counts, fs.hist_edges,
                                 highlight_min=highlight_min)

        if hist:
            print(f'  {t.cyan}{title} Distribution:{t.reset}')
            print(*hist, '', sep='\n')

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

        # Reduce the smallest h together with the element holding it
        eidx = self.stats['min_h_eidx']
        cand = (float(self.stats['min_h']), self.stats['min_h_etype'] or '',
                -1 if eidx is None else int(eidx))
        h, etype, eidx = comm.allreduce(cand, op=mpi.MIN)

        self.stats['min_h'] = h
        self.stats['min_h_etype'] = etype or None
        self.stats['min_h_eidx'] = None if eidx < 0 else eidx

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
        self.worst_h = self._gather_worst(
            lambda r: np.min(r.metrics.h, axis=0), n_worst
        )
        if self.nsr_thresh > 0:
            self.worst_nsr = self._gather_worst(lambda r: r.max_nsr, n_worst,
                                                minimise=False)
        else:
            self.worst_nsr = []

        return rank == root

    def output_text(self, n_worst):
        w = min(shutil.get_terminal_size().columns, 72)
        t = tty
        s = self.stats

        print(f'{t.bold}Mesh Quality Report{t.reset}', '='*w, '', sep='\n')

        for etype in self.mesh.etypes:
            gi = self.ginfo[etype]
            hdr = (f'{t.bold}Element Type: {etype}{t.reset} '
                   f'({gi['neles']} elements, {gi['ncurved']} curved), '
                   f'p = {self.order}, q = {gi['q']}, '
                   f'nupts = {gi['nupts']}, nmpts = {gi['nmpts']}')
            print(hdr + '\n')

            for f, label in self.statlabels.items():
                if f == 'curved' and not gi['ncurved']:
                    continue

                hl = self._highlight_min(etype, f)
                print(_format_stats(self.fieldstats[etype, f], label,
                                    highlight_min=hl) + '\n')

            # Scaled Jacobian stats filtered to curved elements
            if gi['ncurved']:
                hl_curved = self._highlight_min(etype, 'curved')
                self._print_histogram(self.fieldstats[etype, 'curved'],
                                      self.statlabels['curved'],
                                      highlight_min=hl_curved)

            hl_h = self._highlight_min(etype, 'h')
            self._print_histogram(self.fieldstats[etype, 'h'], 'Mesh Scale',
                                  highlight_min=hl_h)

        # Neighbour size ratio section
        if self.nsr:
            nsr = self.nsr
            print(f'{t.bold}Neighbour Size Ratio{t.reset}\n')

            if nsr.max > 5:
                max_str = f'{t.red}{t.bold}{nsr.max:9.3g}{t.reset}'
            else:
                max_str = f'{nsr.max:9.3g}'
            print(f'  Min: {nsr.min:9.3g}  Max: {max_str}  Mean: '
                  f'{nsr.mean:9.3g} ± {nsr.std:9.3g}\n')

            self._print_histogram(nsr, 'Neighbour Size Ratio')

        print('-'*w, f'{t.bold}Summary{t.reset}', '-'*w, sep='\n')

        def _count(label, val):
            c = t.green if val == 0 else t.red
            return f'  {label}  {c}{val}{t.reset}'

        print(f'  Curved elements:            '
              f'{s['n_curved']} / {s['n_total']}')
        print(_count('Inverted elements (J ≤ 0):  ', s['n_inverted']))
        print(_count('Elements with NaN Jacobian: ', s['n_nan']))
        jl = f'Scaled Jacobian < {self.jac_thresh}:'.ljust(28)
        al = f'Aspect ratio > {self.ar_thresh}:'.ljust(28)
        print(_count(jl, s['n_poor_scaled_jac']))
        print(_count(al, s['n_high_aspect']))
        if self.nsr:
            sl = f'Neighbour size ratio > {self.nsr_thresh}:'.ljust(28)
            print(_count(sl, s['n_high_nsr']))
        print()

        min_csj = s['min_curved_scaled_jac']
        if np.isfinite(min_csj):
            c = t.red if min_csj < self.jac_thresh else t.cyan
            print(f'  {c}Min scaled Jacobian (curved):{t.reset} '
                  f'{min_csj:9.3g}')

        if np.isfinite(s['min_h']):
            dt_factor = s['min_h'] / (2*self.order + 1)
            etype, eidx = s['min_h_etype'], s['min_h_eidx']
            print(f'  {t.cyan}Likely CFL limiting element:{t.reset}'
                  f' {etype} {eidx} (h = {s['min_h']:.3g})')
            print(f'  {t.cyan}Geometric dt factor:{t.reset} '
                  f'h_min/(2p+1) = {dt_factor:9.3g}')

        if self.nsr:
            print(f'  {t.cyan}Max neighbour size ratio:{t.reset} '
                  f'{self.nsr.max:9.3g}')

        # Worst elements
        if n_worst > 0:
            print('='*w)

            pwt = _print_worst_table
            pwt(self.worst_sj, 'Worst Elements by Scaled Jacobian',
                'Scaled J')
            pwt(self.worst_h, 'Smallest Mesh Scale (CFL limiting)', 'h_min')

            if self.worst_nsr:
                pwt(self.worst_nsr, 'Worst Neighbour Size Ratios', 'NSR')

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
        make_inf_none = lambda v: float(v) if np.isfinite(v) else None
        glob = {k: s[k] for k in self.gsums if k != 'n_total'}
        for k in self.gmins + ('min_h',):
            glob[k] = make_inf_none(s[k])

        if np.isfinite(s['min_h']):
            glob['geometric_dt_factor'] = s['min_h'] / (2*self.order + 1)
        else:
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

    def export(self, path):
        mesh, cfg = self.mesh, self.cfg

        # Build file stats record
        fstats = Inifile()
        fstats.set('data', 'prefix', 'quality')

        fields = ['scaled-jacobian', 'mesh-scale', 'aspect-ratio', 'is-curved']
        if self.nsr:
            fields.append('size-ratio')
        fstats.set('data', 'fields', ', '.join(fields))

        # Prepare shapes and field groups
        shapes = {et: (len(fields), r.nupts) for et, r in self.etypes.items()}
        field_groups = {'quality': fields}

        # Create writer
        writer = NativeWriter(mesh, cfg, np.float64, path.parent, path.name,
                              'quality')
        writer.set_shapes_eidxs(shapes, mesh.eidxs, field_groups)

        # Pack data per element type: (neles, nfields, nupts)
        data = {}
        for etype, res in self.etypes.items():
            m = res.metrics
            neles, nupts = res.neles, res.nupts

            # Expand per-element scalars to all solution points
            sj = np.broadcast_to(m.scaled_jac[:, None], (neles, nupts))
            ic = np.broadcast_to(res.curved[:, None], (neles, nupts))

            fields = [sj, m.h.T, m.aspect.T, ic.astype(float)]
            if res.max_nsr is not None:
                msr = res.max_nsr[:, None]
                fields.append(np.broadcast_to(msr, (neles, nupts)))

            data[etype] = {'quality': np.stack(fields, axis=1)}

        # Write
        metadata = {
            'mesh-uuid': mesh.uuid,
            'config': cfg.tostr(),
            'stats': fstats.tostr(),
        }
        writer.write(data, tcurr=0.0, metadata=metadata)


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
        parser.add_argument('--export', type=Path, metavar='FILE',
                            help='export quality fields to .pyfrs file')
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

        if args.export:
            ma.export(args.export)
