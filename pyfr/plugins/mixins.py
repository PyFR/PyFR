import shlex

import numpy as np
from pytools import prefork

from pyfr.inifile import NoOptionError
from pyfr.mpiutil import get_comm_rank_root
from pyfr.plugins.common import region_data, surface_data
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

        # Check that we support this particular system
        if not ('*' in self.systems or intg.system.name in self.systems):
            raise RuntimeError(f'System {intg.system.name} not supported by '
                               f'plugin {self.name}')

        # Check that we support dimensionality of simulation
        if intg.system.ndims not in self.dimensions:
            raise RuntimeError(f'Dimensionality of {intg.system.ndims} not '
                               f'supported by plugin {self.name}')

        self.enabled = self.cfg.getbool(cfgsect, 'enabled', True)


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
