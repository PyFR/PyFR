from pyfr.inifile import Inifile
from pyfr.mpiutil import init_mpi
from pyfr.plugins.base import BaseCLIPlugin
from pyfr.plugins.common import cli_external
from pyfr.plugins.soln.ascent import AscentRenderer
from pyfr.snapshot import from_file as snap_from_file


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

        acfg = Inifile.load(args.cfg)
        acfgsect = args.cfgsect or acfg.sections()[0]

        # Current Ascent renderer + its underlying soln config
        renderer, rcfg = None, None

        for s in args.solns:
            snap = snap_from_file(args.mesh, s)
            if snap.prefix != 'soln':
                raise NotImplementedError(
                    f'cli/ascent does not support {snap.prefix!r}-prefix.')

            # Rebuild the renderer when the underlying solver config changes
            if not renderer or rcfg != snap.config:
                renderer = AscentRenderer(snap.mesh, snap.config, acfgsect,
                                          isrestart=True, acfg=acfg)
                rcfg = snap.config

            renderer.render(snap)
