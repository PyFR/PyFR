import functools as ft
import shlex

import numpy as np
from pytools import prefork

from pyfr.inifile import NoOptionError
from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.quadrules import get_quadrule
from pyfr.regions import parse_region_expr
from pyfr.util import memoize


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


def region_data(cfg, cfgsect, mesh):
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

        # Parse the region expression and obtain the element set
        rgn = parse_region_expr(region)
        eset = rgn.interior_eles(mesh)

        # Ensure the region is not empty
        if not comm.reduce(bool(eset), op=mpi.LOR, root=root) and rank == root:
            raise ValueError(f'Empty region {region}')

        return {etype: np.unique(eidxs).astype(np.int32)
                for etype, eidxs in sorted(eset.items())}


def surface_data(cfg, cfgsect, mesh):
    surf = cfg.get(cfgsect, 'surface')

    comm, rank, root = get_comm_rank_root()

    # Parse the surface expression and obtain the element set
    rgn = parse_region_expr(surf)
    eset = rgn.surface_faces(mesh)

    # Ensure the surface is not empty
    if not comm.reduce(bool(eset), op=mpi.LOR, root=root) and rank == root:
        raise ValueError(f'Empty surface {surf}')

    return {etype: np.unique(eidxs).astype(np.int32)
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
