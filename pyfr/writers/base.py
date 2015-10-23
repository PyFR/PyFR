# -*- coding: utf-8 -*-

from pyfr.inifile import Inifile
from pyfr.readers.native import NativeReader
from pyfr.solvers import BaseSystem
from pyfr.util import subclass_where


class BaseWriter(object):
    """Functionality for post-processing PyFR data to visualisation formats"""

    def __init__(self, args):
        """Loads PyFR mesh and solution files

        A check is made to ensure the solution was computed on the mesh.

        :param args: Command line arguments passed from scripts/postp.py
        :type args: class 'argparse.Namespace'

        """
        self.outf = args.outf

        # Load mesh and solution files
        self.soln = NativeReader(args.solnf)
        self.mesh = NativeReader(args.meshf)

        # Get element types and array shapes
        self.mesh_inf = self.mesh.array_info
        self.soln_inf = self.soln.array_info

        # Dimensions
        self.ndims = next(iter(self.mesh_inf.values()))[1][2]
        self.nvars = next(iter(self.soln_inf.values()))[1][1]

        # Check solution and mesh are compatible
        if self.mesh['mesh_uuid'] != self.soln['mesh_uuid']:
            raise RuntimeError('Solution "%s" was not computed on mesh "%s"' %
                               (args.solnf, args.meshf))

        # Load the config file
        self.cfg = Inifile(self.soln['config'])

        # System and elements classs
        self.systemscls = subclass_where(
            BaseSystem, name=self.cfg.get('solver', 'system')
        )
        self.elementscls = self.systemscls.elementscls
