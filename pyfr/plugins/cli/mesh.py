from collections import namedtuple
import json
from pathlib import Path
import shutil

import numpy as np

from pyfr.inifile import Inifile
from pyfr.mpiutil import autofree, get_comm_rank_root, init_mpi, mpi
from pyfr.plugins.base import BaseCLIPlugin
from pyfr.plugins.common import cli_external
from pyfr.readers.native import NativeReader
from pyfr.shapes import BaseShape
from pyfr.util import subclass_where, subclasses, tty
from pyfr.writers.native import NativeWriter


Metrics = namedtuple('Metrics', 'djac h aspect scaled_jac char_len ploc')
EleInfo = namedtuple(
    'EleInfo', 'neles nupts curved eidxs metrics n_inverted n_nan '
    'n_poor_scaled_jac n_high_aspect max_nsr', defaults=(None,)
)
SRStats = namedtuple('SRStats',
                     'n_high min max mean std hist_counts hist_edges')


def _compute_metrics(ele):
    # Jacobian determinant at solution points
    rcpdjac = ele.rcpdjac_at_np('upts')
    djac = 1.0 / rcpdjac

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

    # Scaled Jacobian: normalise by value at element centroid of ideal shape
    # For simplicity, use min(djac) / max(djac) per element as proxy
    djac_min_per_ele = np.min(djac, axis=0)
    djac_max_per_ele = np.max(djac, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        scaled_jac = np.where(
            djac_max_per_ele > 0,
            djac_min_per_ele / djac_max_per_ele,
            np.nan
        )

    # Volume-based characteristic length
    char_len = np.mean(djac, axis=0) ** (1.0 / ele.ndims)

    # Physical locations
    ploc = ele.ploc_at_np('upts')

    return Metrics(djac, h_min, aspect, scaled_jac, char_len, ploc)


def _find_worst(arr, ploc, eidxs, n=10, minimise=True):
    fill = -np.inf if minimise else np.inf
    order = np.argsort(np.where(np.isnan(arr), fill, arr))
    idxs = order[:n] if minimise else order[:-n-1:-1]

    return [(arr[i], eidxs[i], np.mean(ploc[:, :, i], axis=0)) for i in idxs]


def _print_worst_table(worst, title, col):
    t = tty
    print()
    print(f'{t.bold}{title}:{t.reset}')
    print(f'  {'Type':<8} {'Element':>10} {col:>12} Location')
    print(f'  {'-'*8} {'-'*10} {'-'*12} {'-'*20}')

    for j, (etype, el, val, loc) in enumerate(worst):
        loc_str = ', '.join(f'{c:.3f}' for c in loc)

        if j == 0:
            print(f'{t.red}{t.bold}  {etype:<8} {el:>10} {val:>12.4f} '
                  f'({loc_str}){t.reset}')
        else:
            print(f'  {etype:<8} {el:>10} {val:>12.4f} ({loc_str})')


def _render_histogram(values=None, bins=10, width=30, highlight_min=False,
                      counts=None, edges=None):
    t = tty

    if counts is None:
        values = values[np.isfinite(values)]
        if not len(values):
            return []

        counts, edges = np.histogram(values, bins=bins)
        total = len(values)
    else:
        total = counts.sum()

    if not total:
        return []

    # Skip if range is negligible relative to magnitude
    if np.ptp(edges) < 1e-6 * max(abs(edges[-1]), 1e-10):
        return []

    max_count = counts.max() or 1

    # Compute format spec for consistent formatting
    mag = max(abs(edges[0]), abs(edges[-1]), 1e-10)
    exp = int(np.floor(np.log10(mag)))
    if -3 <= exp <= 4:
        fmt = f'8.{max(0, 3 - exp)}f'
    else:
        fmt = '8.2e'

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


def _format_stats(arr, name, highlight_min=False):
    t = tty
    arr_flat = arr.flatten()
    finite = arr_flat[np.isfinite(arr_flat)]
    if not len(finite):
        return f'  {name}: No valid data'

    mn, mx = np.min(finite), np.max(finite)
    mean, std = np.mean(finite), np.std(finite)

    if highlight_min:
        mn_str = f'{t.red}{t.bold}{mn:9.3g}{t.reset}'
    else:
        mn_str = f'{mn:9.3g}'

    return (f'  {name}:\n'
            f'    Min: {mn_str}  Max: {mx:9.3g}  '
            f'Mean: {mean:9.3g} ± {std:9.3g}')


class _MeshAnalyser:
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
        self.stats = {
            'n_total': 0, 'n_inverted': 0, 'n_nan': 0,
            'n_poor_scaled_jac': 0, 'n_high_aspect': 0, 'n_curved': 0,
            'min_scaled_jac': np.inf, 'min_h': np.inf,
            'min_curved_scaled_jac': np.inf,
            'min_h_etype': None, 'min_h_eidx': None,
        }

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

            self.etypes[etype] = EleInfo(ele.neles, ele.nupts, curved,
                                         mesh.eidxs[etype], m, n_inv, n_nan,
                                         n_pj, n_ha)

            self.stats['n_total'] += ele.neles
            self.stats['n_inverted'] += n_inv
            self.stats['n_nan'] += n_nan
            self.stats['n_poor_scaled_jac'] += n_pj
            self.stats['n_high_aspect'] += n_ha
            self.stats['n_curved'] += n_curved

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
        comm, rank, root = get_comm_rank_root()

        # Per-element characteristic length and worst neighbour ratio
        cl = {et: r.metrics.char_len for et, r in self.etypes.items()}
        nsr = {et: np.ones(r.neles) for et, r in self.etypes.items()}

        def process(con, lcl, rcl):
            face_r = np.maximum(lcl, rcl) / np.minimum(lcl, rcl)
            for et, fi, ei, mask in con.foreach():
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

        self.nsr = self._reduce_nsr(nsr)

    def _reduce_nsr(self, nsr):
        comm, rank, root = get_comm_rank_root()

        local_all = np.concatenate(list(nsr.values()))

        # Global stats via scalar reductions
        lo = comm.allreduce(float(np.min(local_all)), op=mpi.MIN)
        hi = comm.allreduce(float(np.max(local_all)), op=mpi.MAX)
        gn = comm.reduce(len(local_all), root=root)
        gs = comm.reduce(float(np.sum(local_all)), root=root)
        gss = comm.reduce(float(np.sum(local_all**2)), root=root)
        gnh = comm.reduce(int(np.sum(local_all > self.nsr_thresh)), root=root)

        # Distributed histogram with globally consistent edges
        edges = np.linspace(lo, hi, 11)
        counts = comm.reduce(np.histogram(local_all, bins=edges)[0], root=root)

        if rank == root:
            gmean = gs / gn
            gstd = np.sqrt(max(gss / gn - gmean**2, 0))
            return SRStats(gnh, lo, hi, gmean, gstd, counts, edges)

    def _gather_worst(self, get_arr, n, minimise=True):
        comm, rank, root = get_comm_rank_root()

        # For each element type, find the worst n elements
        candidates = []
        for etype, res in self.etypes.items():
            worst = _find_worst(get_arr(res), res.metrics.ploc, res.eidxs, n=n,
                                minimise=minimise)
            candidates.extend((etype, gi, val, loc) for val, gi, loc in worst)

        # Locally sort the candidates
        candidates.sort(key=lambda c: c[2], reverse=not minimise)

        # Gather the top n candidates from each rank to the root rank
        candidates = comm.gather(candidates[:n], root=root)
        if rank == root:
            candidates = [c for cl in candidates for c in cl]
            candidates.sort(key=lambda c: c[2], reverse=not minimise)
            return candidates[:n]
        else:
            return []

    def reduce(self, n_worst=0):
        # Reduce per-rank stats to root via MPI
        comm, rank, root = get_comm_rank_root()

        for key in ['n_total', 'n_inverted', 'n_nan', 'n_poor_scaled_jac',
                    'n_high_aspect', 'n_curved']:
            self.stats[key] = comm.reduce(self.stats[key], root=root)

        for key in ['min_scaled_jac', 'min_h', 'min_curved_scaled_jac']:
            self.stats[key] = comm.reduce(self.stats[key], op=mpi.MIN,
                                          root=root)

        if self.nsr:
            self.stats['n_high_nsr'] = self.nsr.n_high
            self.stats['max_nsr'] = self.nsr.max

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

        for etype, res in self.etypes.items():
            m = res.metrics
            hdr = (f'{t.bold}Element Type: {etype}{t.reset} '
                   f'({res.neles} elements, {res.curved.sum()} curved), '
                   f'order = {self.order}, nupts = {res.nupts}')
            print(hdr + '\n')

            # Check if this element type has the global minimum h
            has_min_h = etype == s['min_h_etype']

            stat_fields = [('scaled_jac', 'Scaled Jacobian', False),
                           ('h', 'Mesh Scale (h)', has_min_h),
                           ('aspect', 'Aspect Ratio', False)]
            for key, name, hl in stat_fields:
                fs = _format_stats(getattr(m, key), name, highlight_min=hl)
                print(fs + '\n')

            # Scaled Jacobian stats filtered to curved elements
            if res.curved.any():
                csj = m.scaled_jac[res.curved]
                hl_curved = np.nanmin(csj) < self.jac_thresh
                fs = _format_stats(csj, 'Scaled Jacobian (curved)',
                                   highlight_min=hl_curved)
                print(fs + '\n')

                chist = _render_histogram(csj, highlight_min=hl_curved)
                if chist:
                    print(f'  {t.cyan}Scaled Jacobian (curved) '
                          f'Distribution:{t.reset}')
                    print(*chist, '', sep='\n')

            h_min = np.min(m.h, axis=0)
            hist = _render_histogram(h_min, highlight_min=has_min_h)
            if hist:
                print(f'  {t.cyan}Mesh Scale Distribution:{t.reset}')
                print(*hist, '', sep='\n')

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

            hist = _render_histogram(counts=nsr.hist_counts,
                                     edges=nsr.hist_edges)
            if hist:
                print(f'  {t.cyan}Neighbour Size Ratio Distribution:{t.reset}')
                print(*hist, '', sep='\n')

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
                  f'{s['max_nsr']:9.3g}')

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
        high_nsr = sr is not None and sr.n_high
        if s['n_inverted'] or s['n_nan']:
            status = 'error'
        elif s['n_poor_scaled_jac'] or s['n_high_aspect'] or high_nsr:
            status = 'warning'
        else:
            status = 'ok'

        # Extract global metrics, converting inf to None for JSON
        make_inf_none = lambda v: float(v) if np.isfinite(v) else None
        glob = {k: s[k] for k in ('n_inverted', 'n_nan', 'n_poor_scaled_jac',
                                  'n_high_aspect', 'n_curved')}
        for k in ('min_scaled_jac', 'min_curved_scaled_jac', 'min_h'):
            glob[k] = make_inf_none(s[k])

        if np.isfinite(s['min_h']):
            glob['geometric_dt_factor'] = s['min_h'] / (2*self.order + 1)
        else:
            glob['geometric_dt_factor'] = None

        if sr:
            glob['n_high_size_ratio'] = sr.n_high
            glob['max_size_ratio'] = sr.max

        output = {
            'status': status, 'order': self.order,
            'element_types': {}, 'global': glob,
        }

        _skip = {'curved', 'eidxs', 'metrics', 'n_high_aspect', 'max_nsr'}
        for etype, res in self.etypes.items():
            m = res.metrics
            etd = {k: v for k, v in res._asdict().items() if k not in _skip}
            etd['n_curved'] = int(res.curved.sum())

            for attr in ('scaled_jac', 'h', 'aspect'):
                etd[f'min_{attr}'] = float(np.nanmin(getattr(m, attr)))
                etd[f'max_{attr}'] = float(np.nanmax(getattr(m, attr)))

            if res.curved.any():
                csj = m.scaled_jac[res.curved]
                etd['min_curved_scaled_jac'] = float(np.nanmin(csj))
                etd['max_curved_scaled_jac'] = float(np.nanmax(csj))

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
