from pyfr.readers.native import NativeReader
from pyfr.util import subclass_where


class BaseWriter:
    def __init__(self, meshf, pname=None, cfg=None):
        # Load the mesh
        self.reader = NativeReader(meshf, pname, construct_con=False)

        # Dimensions
        self.ndims = self.reader.mesh.ndims

        # Additional postproc config sections to merge (if any)
        self._ppcfg = cfg

    def _load_soln(self, solnf):
        from pyfr.solvers.base import BaseSystem

        self.mesh, self.soln = self.reader.load_subset_mesh_soln(solnf)

        # Always use solution config, merge in any user-supplied sections
        self.cfg = self.soln['config']
        if self._ppcfg:
            for sect in self._ppcfg.sections():
                for k, v in self._ppcfg.items(sect):
                    self.cfg.set(sect, k, v)

        self.stats = self.soln['stats']

        # Data file prefix
        self.dataprefix = self.stats.get('data', 'prefix')

        # System and elements classes
        self.systemscls = subclass_where(
            BaseSystem, name=self.cfg.get('solver', 'system')
        )
        self.elementscls = self.systemscls.elementscls
