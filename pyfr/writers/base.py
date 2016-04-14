# -*- coding: utf-8 -*-

from pyfr.inifile import Inifile
from pyfr.readers.native import NativeReader
from pyfr.util import subclass_where


class BaseWriter(object):
    def __init__(self, args):
        from pyfr.solvers.base import BaseSystem

        self.outf = args.outf

        # Load the mesh and solution files
        self.soln = NativeReader(args.solnf)
        self.mesh = NativeReader(args.meshf)

        # Check solution and mesh are compatible
        if self.mesh['mesh_uuid'] != self.soln['mesh_uuid']:
            raise RuntimeError('Solution "%s" was not computed on mesh "%s"' %
                               (args.solnf, args.meshf))

        # Load the configuration and stats files
        self.cfg = Inifile(self.soln['config'])
        self.stats = Inifile(self.soln['stats'])

        # Data file prefix (defaults to soln for backwards compatibility)
        self.dataprefix = self.stats.get('data', 'prefix', 'soln')

        # Get element types and array shapes
        self.mesh_inf = self.mesh.array_info('spt')
        self.soln_inf = self.soln.array_info(self.dataprefix)

        # Dimensions
        self.ndims = next(iter(self.mesh_inf.values()))[1][2]
        self.nvars = next(iter(self.soln_inf.values()))[1][1]

        # System and elements classes
        self.systemscls = subclass_where(
            BaseSystem, name=self.cfg.get('solver', 'system')
        )
        self.elementscls = self.systemscls.elementscls
