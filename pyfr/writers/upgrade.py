from collections import defaultdict
from pathlib import Path

import numpy as np

from pyfr.inifile import Inifile
from pyfr.readers.base import NodalMeshAssembler


def _copy_attrs(src_dset, dst_dset):
    for k, v in src_dset.attrs.items():
        dst_dset.attrs[k] = v


def _upgrade_mesh_v1_to_v2(src, dst):
    codec = [c.decode() for c in src['codec']]

    # Copy everything except eles and version
    for k in src:
        if k not in ('eles', 'version'):
            src.copy(k, dst)

    dst['version'] = 2

    # Read element datasets
    eles = {etype: src[f'eles/{etype}'][:] for etype in src['eles']}

    # Compute element colouring
    NodalMeshAssembler.compute_element_colouring(eles, codec)

    # Write upgraded elements
    g = dst.create_group('eles')
    for etype, edata in eles.items():
        g[etype] = edata
        _copy_attrs(src[f'eles/{etype}'], g[etype])


def _soln_field_groups(stats):
    prefix = stats.get('data', 'prefix')
    fields = stats.get('data', 'fields').split(',')

    if prefix == 'soln':
        pmaps = [('grad_', 'grad'), ('', 'soln')]
    elif prefix == 'tavg':
        pmaps = [('std-fun-avg-', 'fun-avg-std'), ('std-', 'avg-std'),
                 ('fun-avg-', 'fun-avg'), ('avg-', 'avg')]
    else:
        raise RuntimeError(f'Unknown data prefix {prefix!r}')

    fg = defaultdict(list)
    for f in fields:
        fpfx, gname = next((p, g) for p, g in pmaps if f.startswith(p))
        fg[gname].append(f.removeprefix(fpfx))

    if 'grad' in fg:
        ndims = len(fg['grad']) // len(fg['soln'])
        fg['grad'] = list(fg['soln'])
    else:
        ndims = 0

    return prefix, fg, ndims


def _upgrade_soln_v1_to_v2(src, dst):
    stats = Inifile(src['stats'][()].decode())

    prefix, field_groups, ndims = _soln_field_groups(stats)

    # Copy everything except the data group and version
    for k in src:
        if k not in (prefix, 'version'):
            src.copy(k, dst)

    dst['version'] = 2
    g = dst.create_group(prefix)

    # Iterate over datasets in the prefix group
    for dk in src[prefix]:
        sdset = src[f'{prefix}/{dk}']

        # Skip -parts (absorbed into aux.part-id)
        if dk.endswith('-parts'):
            continue

        # Copy -idxs verbatim
        if dk.endswith('-idxs'):
            src.copy(sdset, g, dk)
            continue

        # Read old flat array and matching parts
        data = sdset[:]
        parts = src[f'{prefix}/{dk}-parts'][:]
        fpdtype = data.dtype
        neles, _, nupts = data.shape

        # Build compound dtype from field groups
        groups = []
        for gn, fns in field_groups.items():
            fshape = (ndims, nupts) if gn == 'grad' else (nupts,)
            groups.append((gn, [(fn, fpdtype, fshape) for fn in fns]))
        groups.append(('aux', [('part-id', np.int64)]))
        dtype = np.dtype(groups)

        # Pack into compound array
        arr = np.empty(neles, dtype=dtype)
        fidx = 0
        for gname, fnames in field_groups.items():
            for fn in fnames:
                if gname == 'grad':
                    arr[gname][fn] = data[:, fidx:fidx + ndims, :]
                    fidx += ndims
                else:
                    arr[gname][fn] = data[:, fidx, :]
                    fidx += 1
        arr['aux']['part-id'] = parts

        g[dk] = arr
        _copy_attrs(sdset, g[dk])


def upgrade(src, dst):
    upgraders = {
        (1, '.pyfrm'): _upgrade_mesh_v1_to_v2,
        (1, '.pyfrs'): _upgrade_soln_v1_to_v2,
    }

    ver = src['version'][()]
    ftype = Path(src.filename).suffix
    latest = max(v for v, ft in upgraders if ft == ftype) + 1

    if ver >= latest:
        raise RuntimeError('File is already at the latest version')

    for v in range(ver, latest):
        upgraders[(v, ftype)](src, dst)
