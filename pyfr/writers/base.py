# -*- coding: utf-8 -*-

from pyfr.readers.native import read_pyfr_data
from pyfr.inifile import Inifile


class BaseWriter(object):
    """Functionality for post-processing PyFR data to visualisation formats"""

    def __init__(self, args):
        """Loads PyFR mesh and solution files

        A check is made to ensure the solution was computed on the mesh.

        :param args: Command line arguments passed from scripts/postp.py
        :type args: class 'argparse.Namespace'

        """
        self.args = args
        self.outf = args.outf

        # Load mesh and solution files
        self.soln = read_pyfr_data(args.solnf)
        self.mesh = read_pyfr_data(args.meshf)

        # Get element types and array shapes
        self.mesh_inf = self.mesh.array_info
        self.soln_inf = self.soln.array_info

        # Check solution and mesh are compatible
        if self.mesh['mesh_uuid'] != self.soln['mesh_uuid']:
            raise RuntimeError('Solution "%s" was not computed on mesh "%s"' %
                               (args.solnf, args.meshf))

        # Load config file
        self.cfg = Inifile(self.soln['config'].item().decode())
