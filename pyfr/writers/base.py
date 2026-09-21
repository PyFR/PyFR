import re

from pyfr.readers.native import NativeReader
from pyfr.util import subclass_where


class BaseWriter:
    needs_con = False

    def __init__(self, meshf, pname=None):
        # Load the mesh
        self.reader = NativeReader(meshf, pname, construct_con=self.needs_con)

        ndims = self.ndims = self.reader.mesh.ndims

        if not re.fullmatch(self.dimensions, str(ndims)):
            raise RuntimeError(f'{ndims}D grids not supported')

    def _load_soln(self, solnf):
        from pyfr.plugins.postproc import get_source
        from pyfr.solvers.base import BaseSystem

        self.mesh, self.soln = self.reader.load_subset_mesh_soln(solnf)

        # Load the configuration and stats files
        self.cfg = self.soln.config
        self.stats = self.soln.stats

        # Data source for the file prefix
        self.dataprefix = self.stats.get('data', 'prefix')
        self.source = get_source(self.dataprefix, self.cfg, self.stats,
                                 self.ndims)

        # System and elements classes
        self.systemscls = subclass_where(
            BaseSystem, name=self.cfg.get('solver', 'system')
        )
        self.elementscls = self.systemscls.elementscls
