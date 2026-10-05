import numpy as np

from pyfr.inifile import Inifile
from pyfr.mpiutil import init_mpi
from pyfr.plugins.base import BaseCLIPlugin
from pyfr.plugins.common import cli_external
from pyfr.plugins.postproc import get_source
from pyfr.plugins.soln.ascent import AscentRenderer, face_shape_ops
from pyfr.readers.native import NativeReader
from pyfr.shapes import BaseShape
from pyfr.util import subclass_where


class _CLIAdapter:
    def __init__(self, mesh, soln, acfg, cfgsect):
        self.mesh = mesh
        self._soln = soln
        self.scfg = soln.config
        self.acfg = acfg
        self.cfgsect = cfgsect
        self.dtype = np.float32

        # Overlay the ascent config onto the solution config for postproc
        self.ppcfg = ppcfg = Inifile(self.scfg.tostr())
        for sect in acfg.sections():
            for k, v in acfg.items(sect).items():
                ppcfg.set(sect, k, v)

        prefix = soln.stats.get('data', 'prefix')
        self.source = get_source(prefix, self.scfg, soln.stats, mesh.ndims)

    @property
    def tcurr(self):
        return self._soln.stats.getfloat('solver-time-integrator', 'tcurr')

    @property
    def cycle(self):
        stats = self._soln.stats
        return stats.getint('solver-time-integrator', 'nacptsteps', 0)

    @property
    def has_grads(self):
        return self.source.prefix == 'tavg' or 'grad' in self._soln.groups

    def prepare(self, pp_plugins, grads):
        soln, ndims = self._soln, self.mesh.ndims
        nvars = len(soln.fields)

        # Discard the residual group and, unless needed, the gradient group
        soln.groups &= {'grad'} if grads else set()
        nrows = nvars*(1 + ndims) if 'grad' in soln.groups else nvars
        soln.rownames = soln.rownames[:nrows]
        soln.data = {et: d[:, :nrows] for et, d in soln.data.items()}

        self.source.prepare(self.mesh, soln, pp_plugins)

    def block(self, etype, eidxs, grads):
        soln = self._soln
        block = soln.data[etype][..., eidxs]
        names = self.source.pvar_names(soln.rownames, soln.groups)

        return names, self.source.to_pvars(block, soln.groups)

    def soln_op_vpts(self, etype, divisor):
        meshf = self.mesh.spts[etype]

        shapecls = subclass_where(BaseShape, name=etype)
        shape = shapecls(len(meshf), self.scfg)

        svpts = shapecls.std_ele(divisor)
        mesh_op = shape.sbasis.nodal_basis_at(svpts)
        soln_op = shape.ubasis.nodal_basis_at(svpts)

        vpts = mesh_op @ meshf.reshape(len(meshf), -1)
        vpts = vpts.reshape(-1, *meshf.shape[1:])

        return soln_op, vpts

    def face_soln_op_vpts(self, etype, fidx, divisor):
        nspts = len(self.mesh.spts[etype])
        return face_shape_ops(etype, fidx, divisor, nspts, self.scfg)


class AscentCLIPlugin(BaseCLIPlugin):
    name = 'ascent'

    @classmethod
    def add_cli(cls, parser):
        sp = parser.add_subparsers()

        # Render command
        ap_render = sp.add_parser('render', help='ascent render --help')
        ap_render.set_defaults(process=cls.render_cli)
        ap_render.add_argument('mesh', help='mesh file')
        ap_render.add_argument('solns', nargs='*', help='solution files')
        ap_render.add_argument('cfg', help='ascent config file')
        ap_render.add_argument('--cfgsect', help='ascent config file section')

    @cli_external
    def render_cli(self, args):
        # Initialise MPI
        init_mpi()

        reader = NativeReader(args.mesh)
        acfg = Inifile.load(args.cfg)

        # Default to the first section with any scenes defined
        if (acfgsect := args.cfgsect) is None:
            for s in acfg.sections():
                if acfg.items(s, prefix='scene-'):
                    acfgsect = s
                    break
            else:
                raise ValueError('No section with scenes found; use '
                                 '--cfgsect')

        # Current Ascent render and associated config
        renderer, rcfg = None, None

        # Iterate over the solutions
        for s in args.solns:
            # Open the solution and create an Ascent adapter
            mesh, soln = reader.load_subset_mesh_soln(s)
            adapter = _CLIAdapter(mesh, soln, acfg, acfgsect)

            # See if we need to create a new Ascent renderer
            if not renderer or rcfg != soln.config:
                renderer = AscentRenderer(adapter, isrestart=True)
                rcfg = soln.config

            # Augment the data with any fields the plugins require
            adapter.prepare(renderer.postproc_plugins, renderer.need_grads)

            # Perform the rendering
            renderer.render(adapter)
