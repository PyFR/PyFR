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
from pyfr.readers.native import Connectivity
from pyfr.regions import parse_region_expr
from pyfr.util import first
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


class BasePlugin:
    name = None
    systems = None
    suffix = None
    enabled = True
    nsteps = None
    trigger = None
    trigger_comb = None
    trigger_action = 'activate'
    trigger_activated = False
    trigger_write_name = None
    trigger_fire_name = None

    def __init__(self, intg, cfgsect, suffix=None):
        self.cfg = intg.cfg
        self.cfgsect = cfgsect

        self.suffix = suffix
        sfx = f'-{suffix}' if suffix else ''
        self.sprefix = f'plugins/{self.name}{sfx}'

        self.ndims = intg.system.ndims
        self.nvars = intg.system.nvars

        # Tolerance for time comparisons
        self.tol = 5*intg.dtmin

        # Check that we support this particular system
        if not ('*' in self.systems or intg.system.name in self.systems):
            raise RuntimeError(f'System {intg.system.name} not supported by '
                               f'plugin {self.name}')

        # Check that we support dimensionality of simulation
        if intg.system.ndims not in self.dimensions:
            raise RuntimeError(f'Dimensionality of {intg.system.ndims} not '
                               f'supported by plugin {self.name}')

        self.enabled = self.cfg.getbool(cfgsect, 'enabled', True)

    def __call__(self, intg):
        pass

    def trigger_write(self, intg):
        pass

    def finalise(self, intg):
        pass

    def setup(self, sdata, serialiser):
        pass


class PublishMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        try:
            self._pub_name = self.cfg.get(self.cfgsect, 'publish-as')
        except NoOptionError:
            self._pub_name = None

    def _publish(self, intg, **values):
        if self._pub_name is not None:
            values = {k: float(v) for k, v in values.items()}
            intg.triggers.publish(self._pub_name, intg.tcurr, values)


class BaseSolnPlugin(BasePlugin):
    prefix = 'soln'

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        cfg, s = self.cfg, cfgsect
        optval = lambda k: cfg.get(s, k) if cfg.hasopt(s, k) else None

        # Trigger configuration
        self.trigger_action = cfg.get(s, 'trigger-action', 'activate')
        self.trigger_write_name = optval('trigger-write')
        self.trigger_fire_name = optval('trigger-set')

        # Parse trigger: & for AND, | for OR, or a single name
        trig = optval('trigger')
        if trig is None:
            self.trigger = None
            self.trigger_comb = None
        elif '&' in trig:
            self.trigger = names = [t.strip() for t in trig.split('&')]
            self.trigger_comb = lambda tm: all(tm.active(n) for n in names)
        elif '|' in trig:
            self.trigger = names = [t.strip() for t in trig.split('|')]
            self.trigger_comb = lambda tm: any(tm.active(n) for n in names)
        else:
            self.trigger = names = [trig.strip()]
            self.trigger_comb = lambda tm: tm.active(names[0])

        # Step frequency gating
        self.nsteps = int(v) if (v := optval('nsteps')) else None


class BaseSolverPlugin(BasePlugin):
    prefix = 'solver'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Runtime extern binding
        self._extern_values = {}
        self._extern_binders = []

    def _update_extern_values(self):
        pass

    def _register_externs(self, intg, names, spec='scalar fpdtype_t'):
        self._update_extern_values()

        for eles in intg.system.ele_map.values():
            for name in names:
                eles.set_external(name, spec)

        intg.system.register_kernel_callback(names, self._extern_callback)

    def _extern_callback(self, kern):
        self._extern_binders.append(kern.bind)
        kern.bind(**self._extern_values)

    def bind_externs(self):
        for b in self._extern_binders:
            b(**self._extern_values)


class BaseCLIPlugin:
    name = None

    @classmethod
    def add_cli(cls, parser):
        pass


class BackendMixin:
    def _init_backend(self, intg):
        self.backend = intg.backend
        self._ele_banks = intg.system.ele_banks
        self._grad_banks = intg.system.eles_vect_upts
        self._etype_map = {et: i for i, et in enumerate(intg.system.ele_types)}
        self._eos_mod = first(intg.system.ele_map.values()).eos_kernel_module

    def _make_view(self, mat, eidxs, vshape):
        n = len(eidxs)
        return self.backend.view(
            np.full(n, mat.mid), np.zeros(n, dtype=np.int32),
            eidxs, np.ones(n, dtype=np.int32), vshape=vshape
        )


class PostactionMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.postact = None
        self.postactaid = None
        self.postactmode = None

        if self.cfg.hasopt(self.cfgsect, 'post-action'):
            self.postact = str(self.cfg.getpath(self.cfgsect, 'post-action'))
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


class SurfaceRegionMixin:
    def _surf_region(self, intg):
        con = surface_data(intg.cfg, self.cfgsect, intg.system.mesh)

        # Generate the appropriate metadata arrays
        ele_surface_data = {}
        if con is not None:
            for etype, fidx, eidxs in con.items():
                ele_surface_data[f'{etype}_f{fidx}_idxs'] = eidxs

        return con, ele_surface_data


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
