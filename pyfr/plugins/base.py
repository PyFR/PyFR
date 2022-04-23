# -*- coding: utf-8 -*-

import os
import re
import shlex

import numpy as np
from pytools import prefork

from pyfr.mpiutil import get_comm_rank_root
from pyfr.regions import BoundaryRegion, ConstructiveRegion


def init_csv(cfg, cfgsect, header, *, filekey='file', headerkey='header'):
    # Determine the file path
    fname = cfg.get(cfgsect, filekey)

    # Append the '.csv' extension
    if not fname.endswith('.csv'):
        fname += '.csv'

    # Open for appending
    outf = open(fname, 'a')

    # Output a header if required
    if os.path.getsize(fname) == 0 and cfg.getbool(cfgsect, headerkey, True):
        print(header, file=outf)

    # Return the file
    return outf


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

    def __call__(self, intg):
        pass

    def serialise(self, intg):
        return {}


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

    def __del__(self):
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
            cmdline = shlex.split(self.postact.format(**kwargs))

            # Invoke
            if self.postactmode == 'blocking':
                # Store returning code of the post-action
                # If it is different from zero
                # request intg to abort the computation
                intg.abort |= bool(prefork.call(cmdline))
            else:
                self.postactaid = prefork.call_async(cmdline)


class RegionMixin:
    def __init__(self, intg, *args, **kwargs):
        super().__init__(intg, *args, **kwargs)

        # Region of interest
        region = self.cfg.get(self.cfgsect, 'region', '*')

        # All elements
        if region == '*':
            self._prepare_region_data_all(intg)
        # All elements inside some region
        else:
            # Geometric region
            if '(' in region:
                rgn = ConstructiveRegion(region)
            # Boundary region
            else:
                m = re.match(r'(\w+)(?:\s+\+(\d+))?$', region)
                rgn = BoundaryRegion(m[1], nlayers=int(m[2] or 1))

            eset = rgn.interior_eles(intg.system.mesh, intg.rallocs)
            self._prepare_region_data_eset(intg, eset)

    def _prepare_region_data_all(self, intg):
        self._ele_regions = [(i, etype, slice(None))
                             for i, etype in enumerate(intg.system.ele_types)]
        self._ele_region_data = {}

    def _prepare_region_data_eset(self, intg, eset):
        elemap = intg.system.ele_map

        self._ele_regions, self._ele_region_data = [], {}
        for etype, eidxs in sorted(eset.items()):
            doff = intg.system.ele_types.index(etype)
            darr = np.unique(eidxs).astype(np.int32)

            self._ele_regions.append((doff, etype, darr))
            self._ele_region_data[f'{etype}_idxs'] = darr
