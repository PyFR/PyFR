from collections import defaultdict
import functools as ft
from pathlib import Path
from weakref import WeakValueDictionary

import h5py
import numpy as np

from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.readers.native import Connectivity
from pyfr.regions import parse_region_expr
from pyfr.writers.csv import CSVStream


def cli_external(meth):
    @ft.wraps(meth)
    def newmeth(cls, args):
        return meth(cls(), args)

    return classmethod(newmeth)


def init_csv(cfg, cfgsect, header, *, filekey='file', headerkey='header',
             nflush=10):
    # Determine the file path
    fname = cfg.get(cfgsect, filekey)

    header = header if cfg.getbool(cfgsect, headerkey, True) else None
    nflush = cfg.getint(cfgsect, 'flushsteps', nflush)

    return CSVStream(fname, header=header, nflush=nflush)


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
                eset = rgn.interior_eles(mesh)
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
