import shlex
import subprocess

import numpy as np

from pyfr.mpiutil import get_comm_rank_root
from pyfr.plugins.common import (init_csv, init_hdf5_series, region_data,
                                 surface_data)
from pyfr.util import first


class InSituMixin:
    def __init__(self, intg, cfgsect, suffix=None, **kwargs):
        super().__init__(cfg=intg.cfg, cfgsect=cfgsect, ndims=intg.system.ndims,
                         **kwargs)

        self.suffix = suffix
        sfx = f'-{suffix}' if suffix else ''
        self.sprefix = f'plugins/{self.name}{sfx}'

        self.nvars = intg.system.nvars

        # Tolerance for time comparisons
        self.tol = 5*intg.dtmin

        self.enabled = self.cfg.getbool(cfgsect, 'enabled', True)


class PublishMixin:
    def __init__(self, intg, *args, **kwargs):
        super().__init__(intg, *args, **kwargs)

        # Default the namespace to the unique plugin instance identity
        sfx = f'.{self.suffix}' if self.suffix else ''
        self._pub_name = self.cfg.get(self.cfgsect, 'publish-as',
                                      f'{self.name}{sfx}')

        # Register to reject namespace collisions between instances
        intg.triggers.register_publisher(self._pub_name, self.cfgsect)

    def _publish(self, intg, **values):
        values = {k: float(v) for k, v in values.items()}
        intg.triggers.publish(self._pub_name, intg.tcurr, values)


class SeriesWriterMixin:
    def _init_series(self, intg, fields, pts=None):
        match self.cfg.get(self.cfgsect, 'file-format', 'csv'):
            case 'csv':
                if pts is None:
                    header, nflush = ['t', *fields], 1
                else:
                    header = ['t', *'xyz'[:self.ndims], *fields]
                    nflush = len(pts)

                self._series_csv = init_csv(self.cfg, self.cfgsect,
                                            ','.join(header), nflush=nflush,
                                            isrestart=intg.isrestart)
                self._write = self._write_series_csv
            case 'hdf5':
                self._series_h5 = init_hdf5_series(intg, self.cfg,
                                                   self.cfgsect, fields, pts)
                self._write = self._write_series_hdf5
            case _:
                raise ValueError('Invalid file format')

        self._series_pts = pts
        self._series_nf = len(fields)

    def _write_series_csv(self, t, samps):
        samps = np.reshape(samps, (-1, self._series_nf))

        if self._series_pts is None:
            self._series_csv(t, *samps[0])
        else:
            for ploc, samp in zip(self._series_pts, samps):
                self._series_csv(t, *ploc, *samp)

    def _write_series_hdf5(self, t, samps):
        self._series_h5((t, np.reshape(samps, (-1, self._series_nf))))


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
        self.postactproc = None
        self.postactmode = None

        if self.cfg.hasopt(self.cfgsect, 'post-action'):
            self.postact = str(self.cfg.getpath(self.cfgsect, 'post-action'))
            self.postactmode = self.cfg.get(self.cfgsect, 'post-action-mode',
                                            'blocking')

            if self.postactmode not in {'blocking', 'non-blocking'}:
                raise ValueError('Invalid post action mode')

    def finalise(self, intg):
        super().finalise(intg)

        if getattr(self, 'postactproc', None) is not None:
            self.postactproc.wait()

    def _invoke_postaction(self, intg, **kwargs):
        comm, rank, root = get_comm_rank_root()

        # If we have a post-action and are the root rank then fire it
        if rank == root and self.postact:
            # If a post-action is currently running then wait for it
            if self.postactproc is not None:
                self.postactproc.wait()

            # Prepare the command line
            cmdline = shlex.split(self.postact.format_map(kwargs))

            # Invoke
            if self.postactmode == 'blocking':
                if (status := subprocess.call(cmdline)):
                    intg.plugin_abort(status)
            else:
                self.postactproc = subprocess.Popen(cmdline)


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
