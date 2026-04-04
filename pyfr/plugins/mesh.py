import json
from pathlib import Path
import shutil

import numpy as np

from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root, init_mpi, mpi
from pyfr.plugins.base import BaseCLIPlugin, cli_external
from pyfr.readers.native import NativeReader
from pyfr.shapes import BaseShape
from pyfr.util import subclass_where, subclasses, tty
from pyfr.writers.native import NativeWriter


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

    # Physical locations
    ploc = ele.ploc_at_np('upts')

    return {
        'djac': djac,
        'h': h_min,
        'aspect': aspect,
        'scaled_jac': scaled_jac,
        'ploc': ploc,
    }


def _find_worst(arr, ploc, eidxs, n=10, minimise=True):
    fill = -np.inf if minimise else np.inf
    order = np.argsort(np.where(np.isnan(arr), fill, arr))
    idxs = order[:n] if minimise else order[:-n-1:-1]

    return [(arr[i], eidxs[i], np.mean(ploc[:, :, i], axis=0)) for i in idxs]


def _print_worst_table(all_results, title, col, get_arr, n):
    t = tty
    print()
    print(f'{t.bold}{title}:{t.reset}')
    print(f'  {'Type':<8} {'Element':>10} {col:>12} Location')
    print(f'  {'-'*8} {'-'*10} {'-'*12} {'-'*20}')

    worst_all = []
    for etype, res in all_results.items():
        m = res['metrics']
        worst = _find_worst(get_arr(m), m['ploc'], res['eidxs'], n=n)
        for val, el, loc in worst:
            worst_all.append((val, etype, el, loc))

    vals = np.array([x[0] for x in worst_all])
    ranking = np.argsort(np.where(np.isfinite(vals), vals, -np.inf))

    for j, i in enumerate(ranking[:n]):
        val, etype, el, loc = worst_all[i]
        loc_str = ', '.join(f'{c:.3f}' for c in loc)

        if j == 0:
            print(f'{t.red}{t.bold}  {etype:<8} {el:>10} {val:>12.4f} '
                  f'({loc_str}){t.reset}')
        else:
            print(f'  {etype:<8} {el:>10} {val:>12.4f} ({loc_str})')


def _render_histogram(values, bins=10, width=30, highlight_min=False):
    t = tty
    values = values[np.isfinite(values)]
    if not len(values):
        return []

    # Skip if range is negligible relative to magnitude
    if np.ptp(values) < 1e-6*max(np.abs(values).max(), 1e-10):
        return []

    counts, edges = np.histogram(values, bins=bins)
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
        pct = 100 * c / values.size
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


def _collect_metrics(mesh, elementscls, basismap, cfg, jac_thresh, ar_thresh):
    results = {}
    stats = {
        'n_inverted': 0, 'n_nan': 0, 'n_poor_jac': 0, 'n_high_aspect': 0,
        'n_curved': 0,
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
        counts = {
            'n_inverted': int(np.sum(np.any(m['djac'] <= 0, axis=0))),
            'n_nan': int(np.sum(np.any(np.isnan(m['djac']), axis=0))),
            'n_poor_jac': int(np.sum(m['scaled_jac'] < jac_thresh)),
            'n_high_aspect': int(np.sum(np.any(m['aspect'] > ar_thresh,
                                                axis=0))),
        }

        for key, val in counts.items():
            stats[key] += val

        stats['n_curved'] += n_curved

        # Track global minimums
        stats['min_scaled_jac'] = min(stats['min_scaled_jac'],
                                      np.nanmin(m['scaled_jac']))

        if n_curved:
            csj = np.nanmin(m['scaled_jac'][curved])
            stats['min_curved_scaled_jac'] = min(
                stats['min_curved_scaled_jac'], csj
            )

        h_min_per_ele = np.nanmin(m['h'], axis=0)
        min_h_idx = np.nanargmin(h_min_per_ele)
        min_h_val = h_min_per_ele[min_h_idx]
        if min_h_val < stats['min_h']:
            stats['min_h'] = min_h_val
            stats['min_h_etype'] = etype
            stats['min_h_eidx'] = mesh.eidxs[etype][min_h_idx]

        results[etype] = {
            'neles': ele.neles, 'nupts': ele.nupts,
            'n_curved': n_curved, 'curved': curved,
            'eidxs': mesh.eidxs[etype], 'metrics': m, **counts,
        }

    return results, stats


def _reduce_stats(stats):
    comm, rank, root = get_comm_rank_root()

    for key in ['n_inverted', 'n_nan', 'n_poor_jac', 'n_high_aspect',
                'n_curved']:
        stats[key] = comm.reduce(stats[key], root=root)

    for key in ['min_scaled_jac', 'min_h', 'min_curved_scaled_jac']:
        stats[key] = comm.reduce(stats[key], op=mpi.MIN, root=root)

    return rank == root


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

        parser.set_defaults(process=cls.process_cmd)

    @cli_external
    def process_cmd(self, args):
        from pyfr.solvers.base import BaseSystem

        init_mpi()

        reader = NativeReader(args.mesh, pname=args.pname, construct_con=False)
        mesh = reader.mesh
        cfg = Inifile.load(args.cfg)

        # Override polynomial order from config if provided
        if args.order is not None:
            cfg.set('solver', 'order', args.order)
        order = cfg.getint('solver', 'order')

        basismap = {b.name: b for b in subclasses(BaseShape, just_leaf=True)}
        system = subclass_where(BaseSystem, name=cfg.get('solver', 'system'))

        results, stats = _collect_metrics(mesh, system.elementscls, basismap,
                                          cfg, args.jac_thresh, args.ar_thresh)

        if _reduce_stats(stats):
            if args.json:
                self._output_json(results, stats, order, args.jac_thresh,
                                  args.ar_thresh)
            else:
                self._output_text(results, stats, order, args.worst,
                                  args.jac_thresh, args.ar_thresh)

        if args.export:
            self._export_quality(args.export, mesh, results, cfg)

    def _output_json(self, all_results, stats, order, jac_thresh, ar_thresh):
        # Determine status
        if stats['n_inverted'] > 0 or stats['n_nan'] > 0:
            status = 'error'
        elif stats['n_poor_jac'] > 0 or stats['n_high_aspect'] > 0:
            status = 'warning'
        else:
            status = 'ok'

        # Extract global metrics, converting inf to None for JSON
        min_sj = stats['min_scaled_jac']
        min_h = stats['min_h']
        out_sj = float(min_sj) if np.isfinite(min_sj) else None
        out_h = float(min_h) if np.isfinite(min_h) else None
        dt_factor = min_h / (2*order + 1) if np.isfinite(min_h) else None

        min_csj = stats['min_curved_scaled_jac']
        out_csj = float(min_csj) if np.isfinite(min_csj) else None

        output = {
            'status': status,
            'order': order,
            'element_types': {},
            'global': {
                'n_inverted': stats['n_inverted'],
                'n_nan': stats['n_nan'],
                'n_poor_scaled_jac': stats['n_poor_jac'],
                'n_high_aspect': stats['n_high_aspect'],
                'n_curved': stats['n_curved'],
                'min_scaled_jac': out_sj,
                'min_curved_scaled_jac': out_csj,
                'min_h': out_h,
                'geometric_dt_factor': dt_factor,
            }
        }

        for etype, res in all_results.items():
            m = res['metrics']
            nc = res['n_curved']
            curved = res['curved']

            etd = {
                'neles': res['neles'],
                'nupts': res['nupts'],
                'n_curved': nc,
                'min_scaled_jac': float(np.nanmin(m['scaled_jac'])),
                'max_scaled_jac': float(np.nanmax(m['scaled_jac'])),
                'min_h': float(np.nanmin(m['h'])),
                'max_h': float(np.nanmax(m['h'])),
                'min_aspect': float(np.nanmin(m['aspect'])),
                'max_aspect': float(np.nanmax(m['aspect'])),
                'n_inverted': res['n_inverted'],
                'n_poor_jac': res['n_poor_jac'],
            }

            if nc:
                csj = m['scaled_jac'][curved]
                etd['min_curved_scaled_jac'] = float(np.nanmin(csj))
                etd['max_curved_scaled_jac'] = float(np.nanmax(csj))

            output['element_types'][etype] = etd

        print(json.dumps(output, indent=2))

    def _output_text(self, all_results, stats, order, n_worst, jac_thresh,
                     ar_thresh):
        w = min(shutil.get_terminal_size().columns, 72)
        t = tty

        print(f'{t.bold}Mesh Quality Report{t.reset}', '='*w, '', sep='\n')

        n_total = sum(r['neles'] for r in all_results.values())

        for etype, res in all_results.items():
            m = res['metrics']
            nc = res['n_curved']
            hdr = (f'{t.bold}Element Type: {etype}{t.reset} '
                   f'({res["neles"]} elements, {nc} curved), '
                   f'order = {order}, nupts = {res["nupts"]}')
            print(hdr + '\n')

            # Check if this element type has the global minimum h
            has_min_h = (etype == stats['min_h_etype'])

            stat_fields = [('scaled_jac', 'Scaled Jacobian', False),
                           ('h', 'Mesh Scale (h)', has_min_h),
                           ('aspect', 'Aspect Ratio', False)]
            for key, name, hl in stat_fields:
                print(_format_stats(m[key], name, highlight_min=hl) + '\n')

            # Scaled Jacobian stats filtered to curved elements
            curved = res['curved']
            if nc:
                csj = m['scaled_jac'][curved]
                hl_curved = (np.nanmin(csj) < jac_thresh)
                print(_format_stats(csj, 'Scaled Jacobian (curved)',
                                    highlight_min=hl_curved) + '\n')

                chist = _render_histogram(csj, highlight_min=hl_curved)
                if chist:
                    print(f'  {t.cyan}Scaled Jacobian (curved) '
                          f'Distribution:{t.reset}')
                    print(*chist, '', sep='\n')

            h_min_per_ele = np.min(m['h'], axis=0)
            hist = _render_histogram(h_min_per_ele, highlight_min=has_min_h)
            if hist:
                print(f'  {t.cyan}Mesh Scale Distribution:{t.reset}')
                print(*hist, '', sep='\n')

        print('-'*w, f'{t.bold}Summary{t.reset}', '-'*w, sep='\n')

        def _count(label, val):
            c = t.green if val == 0 else t.red
            return f'  {label}  {c}{val}{t.reset}'

        print(f'  Curved elements:            '
              f'{stats["n_curved"]} / {n_total}')
        print(_count('Inverted elements (J ≤ 0):  ', stats['n_inverted']))
        print(_count('Elements with NaN Jacobian: ', stats['n_nan']))
        print(_count(f'Scaled Jacobian < {jac_thresh}:'.ljust(28),
                     stats['n_poor_jac']))
        print(_count(f'Aspect ratio > {ar_thresh}:'.ljust(28),
                     stats['n_high_aspect']))
        print()

        min_csj = stats['min_curved_scaled_jac']
        if np.isfinite(min_csj):
            c = t.red if min_csj < jac_thresh else t.cyan
            print(f'  {c}Min scaled Jacobian (curved):{t.reset} '
                  f'{min_csj:9.3g}')

        if np.isfinite(stats['min_h']):
            dt_factor = stats['min_h'] / (2*order + 1)
            etype, eidx = stats['min_h_etype'], stats['min_h_eidx']
            print(f'  {t.cyan}Likely CFL limiting element:{t.reset} '
                  f'{etype} {eidx} (h = {stats["min_h"]:.3g})')
            print(f'  {t.cyan}Geometric dt factor:{t.reset} '
                  f'h_min/(2p+1) = {dt_factor:9.3g}')

        # Worst elements
        if n_worst > 0:
            print('='*w)

            pwt = _print_worst_table
            pwt(all_results, 'Worst Elements by Scaled Jacobian',
                'Scaled J', lambda m: m['scaled_jac'], n_worst)
            pwt(all_results, 'Smallest Mesh Scale (CFL limiting)',
                'h_min', lambda m: np.min(m['h'], axis=0), n_worst)

    def _export_quality(self, path, mesh, all_results, cfg):
        # Build file stats record
        fstats = Inifile()
        fstats.set('data', 'prefix', 'quality')
        fstats.set('data', 'fields',
                   'scaled-jacobian, mesh-scale, aspect-ratio, is-curved')

        # Prepare shapes and field groups
        fields = ['scaled-jacobian', 'mesh-scale', 'aspect-ratio',
                  'is-curved']
        shapes = {et: (len(fields), r['nupts'])
                  for et, r in all_results.items()}
        field_groups = {'quality': fields}

        # Create writer
        writer = NativeWriter(mesh, cfg, np.float64, path.parent, path.name,
                              'quality')
        writer.set_shapes_eidxs(shapes, mesh.eidxs, field_groups)

        # Pack data per element type: (neles, nfields, nupts)
        data = {}
        for etype, res in all_results.items():
            m = res['metrics']
            neles, nupts = res['neles'], res['nupts']

            # Expand per-element scalars to all solution points
            sj = np.broadcast_to(m['scaled_jac'][:, None], (neles, nupts))
            ic = np.broadcast_to(res['curved'][:, None].astype(float),
                                 (neles, nupts))

            quality = np.stack([sj, m['h'].T, m['aspect'].T, ic], axis=1)
            data[etype] = {'quality': quality}

        # Write
        metadata = {
            'mesh-uuid': mesh.uuid,
            'config': cfg.tostr(),
            'stats': fstats.tostr()
        }
        writer.write(data, tcurr=0.0, metadata=metadata)
