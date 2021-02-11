# -*- coding: utf-8 -*-

from collections import defaultdict
import os
import shlex

import numpy as np
from pytools import prefork

from pyfr.mpiutil import get_comm_rank_root
from pyfr.writers.native import NativeWriter


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


class BasePlugin(object):
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


class PostactionMixin(object):
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
                intg.abort |= not prefork.call(cmdline)
            else:
                self.postactaid = prefork.call_async(cmdline)


class RegionMixin(object):
    def _init_writer_for_region(self, intg, nout, prefix, *, fpdtype=None):
        # Base output directory and file name
        basedir = self.cfg.getpath(self.cfgsect, 'basedir', '.', abs=True)
        basename = self.cfg.get(self.cfgsect, 'basename')

        # Data type
        if fpdtype is None:
            fpdtype = intg.backend.fpdtype
        elif fpdtype == 'single':
            fpdtype = np.float32
        elif fpdtype == 'double':
            fpdtype = np.float64
        else:
            raise ValueError('Invalid floating point data type')

        # Region of interest
        region = self.cfg.get(self.cfgsect, 'region', '*')

        # All elements
        if region == '*':
            mdata = self._prepare_mdata_all(intg, fpdtype, nout, prefix)
            self._add_region_data = self._add_region_data_all
        # All elements inside a box
        elif '(' in region or '[' in region:
            box = self.cfg.getliteral(self.cfgsect, 'region')
            mdata = self._prepare_mdata_box(intg, fpdtype, nout, prefix, *box)
            self._add_region_data = self._add_region_data_subset
        # All elements on a boundary
        else:
            mdata = self._prepare_mdata_bcs(intg, fpdtype, nout, prefix,
                                            region)
            self._add_region_data = self._add_region_data_subset

        # Construct the file writer
        return NativeWriter(intg, mdata, basedir, basename)

    def _prepare_mdata_all(self, intg, fpdtype, nout, prefix):
        self._ele_regions = [(i, slice(None))
                             for i in range(len(intg.system.ele_types))]

        # Element info
        einfo = zip(intg.system.ele_types, intg.system.ele_shapes)

        # Output metadata
        return [(f'{prefix}_{etype}', (nupts, nout, neles), fpdtype)
                for etype, (nupts, nvars, neles) in einfo]

    def _prepare_mdata_box(self, intg, fpdtype, nout, prefix, x0, x1):
        eset = {}

        for etype in intg.system.ele_types:
            pts = intg.system.mesh[f'spt_{etype}_p{intg.rallocs.prank}']
            pts = np.moveaxis(pts, 2, 0)

            # Determine which points are inside the box
            inside = np.ones(pts.shape[1:], dtype=np.bool)
            for l, p, u in zip(x0, pts, x1):
                inside &= (l <= p) & (p <= u)

            if np.sum(inside):
                eset[etype] = np.any(inside, axis=0).nonzero()[0]

        return self._prepare_eset(intg, fpdtype, nout, prefix, eset)

    def _prepare_mdata_bcs(self, intg, fpdtype, nout, prefix, bcname):
        comm, rank, root = get_comm_rank_root()

        # Get the mesh and prepare the element set dict
        mesh = intg.system.mesh
        eset = defaultdict(list)

        # Boundary of interest
        bc = f'bcon_{bcname}_p{intg.rallocs.prank}'

        # Ensure the boundary exists
        bcranks = comm.gather(bc in mesh, root=root)
        if rank == root and not any(bcranks):
            raise ValueError(f'Boundary {bcname} does not exist')

        if bc in mesh:
            # Determine which of our elements are on the boundary
            for etype, eidx in mesh[bc][['f0', 'f1']].astype('U4,i4'):
                eset[etype].append(eidx)

        return self._prepare_eset(intg, fpdtype, nout, prefix, eset)

    def _prepare_eset(self, intg, fpdtype, nout, prefix, eset):
        elemap = intg.system.ele_map

        mdata, self._ele_regions = [], []
        for etype, eidxs in sorted(eset.items()):
            neidx = len(eidxs)
            shape = (elemap[etype].nupts, nout, neidx)

            mdata.append((f'{prefix}_{etype}', shape, fpdtype))
            mdata.append((f'{prefix}_{etype}_idxs', (neidx,), np.int32))

            doff = intg.system.ele_types.index(etype)
            darr = np.unique(eidxs).astype(np.int32)

            self._ele_regions.append((doff, darr))

        return mdata

    def _add_region_data_all(self, data):
        return data

    def _add_region_data_subset(self, data):
        ndata = []

        for (i, rgn), darr in zip(self._ele_regions, data):
            ndata.append(darr)
            ndata.append(rgn)

        return ndata
