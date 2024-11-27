from collections import defaultdict
import functools as ft
from pathlib import Path
import shlex
from weakref import WeakValueDictionary

import h5py
import numpy as np
from pytools import prefork

from pyfr.inifile import NoOptionError
from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.cache import memoize
from pyfr.quadrules import get_quadrule
from pyfr.regions import parse_region_expr


def cli_external(meth):
    @ft.wraps(meth)
    def newmeth(cls, args):
        return meth(cls(), args)

    return classmethod(newmeth)


def init_csv(cfg, cfgsect, header, *, filekey='file', headerkey='header'):
    # Determine the file path
    fname = cfg.get(cfgsect, filekey)

    # Append the '.csv' extension
    if not fname.endswith('.csv'):
        fname += '.csv'

    # Open for appending
    outf = open(fname, 'a')

    # Output a header if required
    if outf.tell() == 0 and cfg.getbool(cfgsect, headerkey, True):
        print(header, file=outf)

    # Return the file
    return outf


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
                for etype, eidxs in sorted(eset.items())}


def surface_data(cfg, cfgsect, mesh):
    surf = cfg.get(cfgsect, 'surface')

    comm, rank, root = get_comm_rank_root()

    # Parse the surface expression and obtain the element set
    rgn = parse_region_expr(surf, mesh.raw.get('regions'))
    eset = rgn.surface_faces(mesh)

    # Ensure the surface is not empty
    if not comm.reduce(bool(eset), op=mpi.LOR, root=root) and rank == root:
        raise ValueError(f'Empty surface {surf}')

    return {etype: np.unique(eidxs).astype(int)
            for etype, eidxs in sorted(eset.items())}


class BasePlugin:
    name = None
    systems = None
    formulations = None

    def __init__(self, intg, cfgsect, suffix=None):
        self.cfg = intg.cfg
        self.cfgsect = cfgsect

        self.suffix = suffix

        self.ndims = intg.system.ndims
        self.nvars = intg.system.nvars

        # Tolerance for time comparisons
        self.tol = 5*intg.dtmin

        # Check that we support this particular system
        if not ('*' in self.systems or intg.system.name in self.systems):
            raise RuntimeError(f'System {intg.system.name} not supported by '
                               f'plugin {self.name}')

        # Check that we support this particular integrator formulation
        if intg.formulation not in self.formulations:
            raise RuntimeError(f'Formulation {intg.formulation} not '
                               f'supported by plugin {self.name}')

        # Check that we support dimensionality of simulation
        if intg.system.ndims not in self.dimensions:
            raise RuntimeError(f'Dimensionality of {intg.system.ndims} not '
                               f'supported by plugin {self.name}')

    def __call__(self, intg):
        pass

    def serialise(self, intg):
        return {}

    def finalise(self, intg):
        pass


class BaseSolnPlugin(BasePlugin):
    prefix = 'soln'


class BaseSolverPlugin(BasePlugin):
    prefix = 'solver'


class BaseCLIPlugin:
    name = None

    @classmethod
    def add_cli(cls, parser):
        pass


class PostactionMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.postact = None
        self.postactaid = None
        self.postactmode = None

        if self.cfg.hasopt(self.cfgsect, 'post-action'):
            self.postact = self.cfg.getpath(self.cfgsect, 'post-action')
            self.postactmode = self.cfg.get(self.cfgsect, 'post-action-mode',
                                            'blocking')

            if self.postactmode not in {'blocking', 'non-blocking'}:
                raise ValueError('Invalid post action mode')

    def finalise(self, intg):
        super().finalise(intg)

        if getattr(self, 'postactaid', None) is not None:
            prefork.wait(self.postactaid)

    def _invoke_postaction(self, intg, **kwargs):
        comm, rank, root = get_comm_rank_root()

        # If we have a post-action and are the root rank then fire it
        if rank == root and self.postact:
            # If a post-action is currently running then wait for it
            if self.postactaid is not None:
                prefork.wait(self.postactaid)

            # Prepare the command line
            cmdline = shlex.split(self.postact.format_map(kwargs))

            # Invoke
            if self.postactmode == 'blocking':
                if (status := prefork.call(cmdline)):
                    intg.plugin_abort(status)
            else:
                self.postactaid = prefork.call_async(cmdline)


class RegionMixin:
    def __init__(self, intg, *args, **kwargs):
        super().__init__(intg, *args, **kwargs)

        # Parse the region
        ridxs = region_data(self.cfg, self.cfgsect, intg.system.mesh)

        # Generate the appropriate metadata arrays
        self._ele_regions, self._ele_region_data = [], {}
        for etype, eidxs in ridxs.items():
            doff = intg.system.ele_types.index(etype)
            self._ele_regions.append((doff, etype, eidxs))

            # Obtain the global element numbers
            geidxs = intg.system.mesh.eidxs[etype][eidxs]
            self._ele_region_data[etype] = geidxs


class SurfaceMixin:
    def _surf_region(self, intg):
        # Parse the region
        sidxs = surface_data(intg.cfg, self.cfgsect, intg.system.mesh)

        # Generate the appropriate metadata arrays
        ele_surface, ele_surface_data = [], {}
        for (etype, face), eidxs in sidxs.items():
            doff = intg.system.ele_types.index(etype)
            ele_surface.append((doff, etype, face, eidxs))

            if not isinstance(eidxs, slice):
                ele_surface_data[f'{etype}_f{face}_idxs'] = eidxs
        return ele_surface, ele_surface_data

    @memoize
    def _surf_quad(self, itype, proj, flags=''):
        # Obtain quadrature info
        rname = self.cfg.get(f'solver-interfaces-{itype}', 'flux-pts')

        # Quadrature rule (default to that of the solution points)
        qrule = self.cfg.get(self.cfgsect, f'quad-pts-{itype}', rname)
        try:
            qdeg = self.cfg.getint(self.cfgsect, f'quad-deg-{itype}')
        except NoOptionError:
            qdeg = self.cfg.getint(self.cfgsect, 'quad-deg')

        # Get the quadrature rule
        q = get_quadrule(itype, qrule, qdeg=qdeg, flags=flags)

        # Project its points onto the provided surface
        pts = np.atleast_2d(q.pts.T)
        return np.vstack(np.broadcast_arrays(*proj(*pts))).T, q.wts


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
