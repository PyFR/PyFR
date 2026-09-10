from collections import defaultdict
import functools as ft
from pathlib import Path
from weakref import WeakValueDictionary

import h5py
import numpy as np

from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.readers.native import Connectivity
from pyfr.regions import parse_region_expr
from pyfr.util import CSVStream, subclass_where


def cli_external(meth):
    @ft.wraps(meth)
    def newmeth(cls, args):
        return meth(cls(), args)

    return classmethod(newmeth)


def get_elementscls(cfg):
    from pyfr.solvers.base import BaseSystem

    systemcls = subclass_where(BaseSystem, name=cfg.get('solver', 'system'))
    return systemcls.elementscls


def init_csv(cfg, cfgsect, header, *, filekey='file', headerkey='header',
             nflush=10, isrestart=True):
    # Determine the file path
    fname = cfg.get(cfgsect, filekey)

    header = header if cfg.getbool(cfgsect, headerkey, True) else None
    nflush = cfg.getint(cfgsect, 'flushsteps', nflush)

    # Fresh runs may clear out any existing data
    reset = cfg.getbool(cfgsect, 'file-reset', False) and not isrestart

    return CSVStream(fname, header=header, nflush=nflush, reset=reset)


def open_hdf5_a(path):
    path = Path(path).absolute()

    try:
        pool = open_hdf5_a.pool
    except AttributeError:
        pool = open_hdf5_a.pool = WeakValueDictionary()

    try:
        return pool[path]
    except KeyError:
        f = pool[path] = h5py.File(path, 'a', libver='latest')

        return f


def region_data(cfg, cfgsect, mesh, *, rtype=None):
    comm, rank, root = get_comm_rank_root()
    region = cfg.get(cfgsect, 'region', '*')

    # Determine the element types in our partition
    etypes = list(mesh.spts)

    # All elements
    if region == '*':
        return {etype: slice(None) for etype in etypes}
    # All elements inside some region
    else:
        comm, rank, root = get_comm_rank_root()

        # Parse the region expression
        rgn = parse_region_expr(region, mesh.raw.get('regions'))

        # Obtain the element set
        match rtype or cfg.get(cfgsect, 'region-type', 'volume'):
            case 'volume':
                eset = rgn.region_eles(mesh)
            case 'surface':
                eset = defaultdict(list)
                for (etype, fidx), eidxs in rgn.surface_faces(mesh).items():
                    eset[etype].extend(eidxs)
            case _:
                raise ValueError('Invalid region type')

        # Ensure the region is not empty
        if not comm.reduce(bool(eset), op=mpi.LOR, root=root) and rank == root:
            raise ValueError(f'Empty region {region}')

        # If requested, expand the region
        if nexpand := cfg.getint(cfgsect, 'region-expand', 0):
            eset = rgn.expand(mesh, eset, nexpand)

        return {etype: np.unique(eidxs).astype(int)
                for etype, eidxs in sorted(eset.items())
                if len(eidxs)}


def surface_data(cfg, cfgsect, mesh):
    surf = cfg.get(cfgsect, 'surface')

    comm, rank, root = get_comm_rank_root()

    # Parse the surface expression and obtain the element set
    rgn = parse_region_expr(surf, mesh.raw.get('regions'))
    eset = rgn.surface_faces(mesh)

    # Ensure the surface is not empty
    if not comm.reduce(bool(eset), op=mpi.LOR, root=root) and rank == root:
        raise ValueError(f'Empty surface {surf}')

    if not eset:
        return None

    # Build a Connectivity from the surface face data
    cidxmap, cidx_a, eidx_a = {}, [], []
    for cidx, (k, v) in enumerate(sorted(eset.items())):
        cidxmap[cidx] = k
        eidxs = np.unique(v)
        cidx_a.append(np.broadcast_to(cidx, len(eidxs)))
        eidx_a.append(eidxs)

    return Connectivity(np.concatenate(cidx_a), np.concatenate(eidx_a),
                        cidxmap)


def init_hdf5_series(intg, cfg, cfgsect, fields, pts=None):
    outf = open_hdf5_a(cfg.get(cfgsect, 'file'))
    dname = cfg.get(cfgsect, 'file-dataset')
    label = ','.join(fields)
    nvars = len(fields)
    npts = 1 if pts is None else len(pts)

    # Fresh runs may clear out any existing series data
    if (cfg.getbool(cfgsect, 'file-reset', False) and not intg.isrestart and
        dname in outf):
        del outf[dname]

    # Each record is a time and the associated samples
    dtype = np.dtype([('t', float), ('samples', float, (npts, nvars))])

    # Cap chunks at 128 records or ~1 MiB, whichever is smaller
    chunk = max(1, min(128, 1024**2 // dtype.itemsize))

    if dname in outf:
        d = outf[dname]

        if d.dtype != dtype or d.attrs['fields'] != label:
            raise ValueError(f'Dataset {dname} exists with different '
                             'fields; use file-reset or a new dataset')

        # Series data is tied to the mesh through its UUID
        if d.attrs['mesh-uuid'] != intg.mesh_uuid:
            raise ValueError('Mesh does not match the existing series')

        # Ensure any point sets are compatible
        if pts is not None and not np.allclose(d.attrs['pts'], pts):
            raise ValueError('Inconsistent sample points')
    else:
        d = outf.create_dataset(dname, (0,), dtype, chunks=(chunk,),
                                maxshape=(None,))

        d.attrs['fields'] = label
        d.attrs['mesh-uuid'] = intg.mesh_uuid
        if pts is not None:
            d.attrs['pts'] = pts

    # Maintain the configuration chain for offline post-processing
    for k, v in intg.cfgmeta.items():
        d.attrs[k] = v

    return DatasetAppender(d)


class DatasetAppender:
    def __init__(self, dset, flush=None, swmr=True):
        self.dset = dset
        self.file = dset.file
        self.swmr = swmr

        flush = flush or dset.chunks[0]

        self._buf = np.empty((flush, *dset.shape[1:]), dtype=dset.dtype)
        self._i = 0

    def __del__(self):
        self.flush()

    def __call__(self, v):
        self._buf[self._i] = v
        self._i += 1

        if self._i == len(self._buf):
            self.flush()

    def flush(self):
        if self._i:
            n = len(self.dset)

            self.dset.resize((n + self._i, *self.dset.shape[1:]))
            self.dset[n:] = self._buf[:self._i]
            self.dset.flush()

            if self.swmr and not self.file.swmr_mode:
                self.file.swmr_mode = True

            self._i = 0
